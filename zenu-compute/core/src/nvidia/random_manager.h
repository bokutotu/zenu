#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <atomic>
#include "zenu_compute.h"

class RandManager {
public:
    RandManager()
        : m_gen(nullptr),
          m_inited(false),
          m_device_id(-1),
          m_seed(1234ULL)
    {
        // コンストラクタではまだ生成しない（lazy init）。
    }

    ~RandManager() {
        // 破棄
        if (m_gen != nullptr) {
            curandDestroyGenerator(m_gen);
            m_gen = nullptr;
        }
        m_inited = false;
    }

    /**
     * @brief 正規分布乱数を生成して GPU メモリ(dst) に書き込む。
     * 
     * @param dst         GPUメモリ上の出力先ポインタ (float* or double*)
     * @param num_elements 生成する乱数の個数
     * @param mean        平均
     * @param stddev      標準偏差 (0以上)
     * @param data_type   f32 or f64
     * @param device_id   生成に使用するGPU ID
     */
    ZenuStatus generate_normal(
        void* dst,
        int num_elements,
        float mean,
        float stddev,
        ZenuDataType data_type,
        int device_id
    )
    {
        if (!dst) {
            return InvalidArgument;
        }
        if (num_elements <= 0) {
            return InvalidArgument;
        }
        if (stddev < 0.0f) {
            return InvalidArgument;
        }

        // Lazy-init (必要に応じてGPUを切り替える)
        ZenuStatus st = ensure_initialized(device_id);
        if (st != Success) {
            return st;
        }

        // 乱数生成 (cuRAND)
        if (data_type == f32) {
            curandStatus_t err = curandGenerateNormal(
                m_gen,
                static_cast<float*>(dst),
                static_cast<size_t>(num_elements),
                mean,
                stddev
            );
            if (err != CURAND_STATUS_SUCCESS) {
                return CurandError;
            }
        }
        else if (data_type == f64) {
            curandStatus_t err = curandGenerateNormalDouble(
                m_gen,
                static_cast<double*>(dst),
                static_cast<size_t>(num_elements),
                static_cast<double>(mean),
                static_cast<double>(stddev)
            );
            if (err != CURAND_STATUS_SUCCESS) {
                return CurandError;
            }
        }
        else {
            return InvalidArgument;
        }

        // 同期（簡易のため、ここでは同期してエラーをチェック）
        cudaError_t cudaErr = cudaDeviceSynchronize();
        if (cudaErr != cudaSuccess) {
            return CudaError;
        }

        return Success;
    }

    /**
     * @brief 一様分布乱数を生成して GPU メモリ(dst) に書き込む。
     *        cuRAND では curandGenerateUniform / curandGenerateUniformDouble を使用。
     * 
     * @param dst         GPUメモリ上の出力先ポインタ (float* or double*)
     * @param num_elements 生成する乱数の個数
     * @param low         一様分布の下限
     * @param high        一様分布の上限
     * @param data_type   f32 or f64
     * @param device_id   生成に使用するGPU ID
     */
    ZenuStatus generate_uniform(
        void* dst,
        int num_elements,
        float low,
        float high,
        ZenuDataType data_type,
        int device_id
    )
    {
        if (!dst) {
            return InvalidArgument;
        }
        if (num_elements <= 0) {
            return InvalidArgument;
        }
        if (low > high) {
            return InvalidArgument;
        }

        ZenuStatus st = ensure_initialized(device_id);
        if (st != Success) {
            return st;
        }

        double range = static_cast<double>(high) - static_cast<double>(low);

        // cuRAND で [0,1) → そのままだと 0～1 の乱数ができるので，
        // 生成後に「x = x * range + low」というカーネルを呼ぶか，
        // あるいは軽量に実装するには下記のように2段階に分けます。
        // 1) generateUniform(...) で [0,1) 乱数を作る
        // 2) カーネルで (value * range + low) に変換
        // 
        // ここでは簡易のために(1)だけ行い、ホスト側or別カーネルで2)を行う例。
        // もし [low, high] をGPU上で直接生成したい場合はカスタムカーネルが必要。
        // ※ cuRAND に "generateUniformRange(gen, ptr, num, low, high)" は無い。
        
        // ただし cuRAND には [0,1) → generateUniform() か generateUniformDouble() はある。

        if (data_type == f32) {
            curandStatus_t err = curandGenerateUniform(
                m_gen,
                static_cast<float*>(dst),
                static_cast<size_t>(num_elements)
            );
            if (err != CURAND_STATUS_SUCCESS) {
                return CurandError;
            }

            // 以下、(value = value * range + low) のカーネルを実行する
            // 簡易的にここで同期し、別途カーネルを呼ぶ
            cudaError_t cudaErr = cudaDeviceSynchronize();
            if (cudaErr != cudaSuccess) {
                return CudaError;
            }

            // range, low を適用するためのカーネル（添付サンプル下に記述）を呼ぶ
            dim3 block(256);
            dim3 grid((num_elements + block.x - 1) / block.x);
            scale_uniform_f32_kernel<<<grid, block>>>(
                static_cast<float*>(dst),
                num_elements,
                static_cast<float>(range),
                low
            );
            cudaErr = cudaDeviceSynchronize();
            if (cudaErr != cudaSuccess) {
                return CudaError;
            }
        }
        else if (data_type == f64) {
            curandStatus_t err = curandGenerateUniformDouble(
                m_gen,
                static_cast<double*>(dst),
                static_cast<size_t>(num_elements)
            );
            if (err != CURAND_STATUS_SUCCESS) {
                return CurandError;
            }

            cudaError_t cudaErr = cudaDeviceSynchronize();
            if (cudaErr != cudaSuccess) {
                return CudaError;
            }

            // scale_kernel for double
            dim3 block(256);
            dim3 grid((num_elements + block.x - 1) / block.x);
            scale_uniform_f64_kernel<<<grid, block>>>(
                static_cast<double*>(dst),
                num_elements,
                static_cast<double>(range),
                static_cast<double>(low)
            );
            cudaErr = cudaDeviceSynchronize();
            if (cudaErr != cudaSuccess) {
                return CudaError;
            }
        }
        else {
            return InvalidArgument;
        }

        return Success;
    }

    /**
     * @brief シード値を変更したい場合
     */
    void set_seed(unsigned long long seed_val) {
        m_seed = seed_val;
        // 再初期化フラグを立てるか、あるいはすぐに更新するかは設計による
        // ここでは一旦Destroyし、再度Createするという方法でもOK
        // 例：
        if (m_gen != nullptr) {
            curandDestroyGenerator(m_gen);
            m_gen = nullptr;
            m_inited = false; 
        }
    }

private:
    //==============================
    // メンバ変数
    //==============================
    curandGenerator_t m_gen;
    bool m_inited;
    int  m_device_id;
    unsigned long long m_seed;

    //==============================
    // ensure_initialized
    //==============================
    ZenuStatus ensure_initialized(int device_id) {
        if (m_inited) {
            // すでに初期化済み
            // 別の device_id を要求された場合、どうするか？
            // ここでは「異なるdevice_idが来ても無視する」実装
            return Success;
        }

        // 初期化処理
        cudaError_t cudaErr = cudaSetDevice(device_id);
        if (cudaErr != cudaSuccess) {
            return CudaError;
        }

        curandStatus_t curandErr =
            curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        if (curandErr != CURAND_STATUS_SUCCESS) {
            return CurandError;
        }

        // シード設定
        curandErr = curandSetPseudoRandomGeneratorSeed(m_gen, m_seed);
        if (curandErr != CURAND_STATUS_SUCCESS) {
            curandDestroyGenerator(m_gen);
            m_gen = nullptr;
            return CurandError;
        }

        // 成功したのでフラグを立てる
        m_inited = true;
        m_device_id = device_id;
        return Success;
    }

    //==============================
    // カーネル: [0,1) -> [low, low+range] への変換 (float版)
    //==============================
    static __global__ void scale_uniform_f32_kernel(
        float* arr,
        int n,
        float range,
        float low
    )
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            float val = arr[idx];
            val = val * range + low;
            arr[idx] = val;
        }
    }

    //==============================
    // カーネル: [0,1) -> [low, low+range] (double版)
    //==============================
    static __global__ void scale_uniform_f64_kernel(
        double* arr,
        int n,
        double range,
        double low
    )
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            double val = arr[idx];
            val = val * range + low;
            arr[idx] = val;
        }
    }
};


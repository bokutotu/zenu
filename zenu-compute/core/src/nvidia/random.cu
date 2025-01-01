#include "zenu_compute_random.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <mutex> // std::mutex
#include <atomic>

//==============================
// RandGPUContext
//==============================
typedef struct {
    bool inited;
    int  device_id;
    unsigned long long seed;
    curandGenerator_t  gen;
} RandGPUContext;

// 同じように16スロット用意 (簡易例)
static RandGPUContext g_contexts[16];
static std::once_flag g_initOnce;
static std::mutex g_contextMutex;

//----------------------------------------
// initRandGPUContexts(): 全要素を初期化
//----------------------------------------
static void initRandGPUContexts()
{
    for (int i = 0; i < 16; i++) {
        g_contexts[i].inited    = false;
        g_contexts[i].device_id = -1;
        g_contexts[i].seed      = 0ULL;
        g_contexts[i].gen       = nullptr;
    }
}

/**
 * @brief getRandGPUContext
 *        device_id に対応する RandGPUContext* を返す。
 *        なければ空きスロットを確保する。
 */
static RandGPUContext* getRandGPUContext(int device_id)
{
    std::call_once(g_initOnce, [](){
        initRandGPUContexts();
    });

    std::lock_guard<std::mutex> lock(g_contextMutex);

    // 既存コンテキストを探す
    for (int i=0; i<16; i++){
        if (g_contexts[i].inited && g_contexts[i].device_id == device_id) {
            return &g_contexts[i];
        }
    }
    // 未使用スロットを探す
    for (int i=0; i<16; i++){
        if (!g_contexts[i].inited) {
            // 使う
            g_contexts[i].inited    = false; // まだ完全初期化していない
            g_contexts[i].device_id = device_id;
            return &g_contexts[i];
        }
    }
    return nullptr; // スロットが無い
}

/**
 * @brief ensureInitedGPUContext
 *        RandGPUContext を初期化(初回のみ)。
 *        ただしシード更新をしたい場合は毎回 setPseudoRandomGeneratorSeed する。
 */
static ZenuStatus ensureInitedGPUContext(RandGPUContext* ctx, unsigned long long seed)
{
    if (!ctx) return DeviceError;

    // もし異なる seed が来た場合、毎回 setPseudoRandomGeneratorSeed を呼びたい。
    // あるいは seed が変わるたびに generator を destroy & create してもよい。
    // ここでは "初回だけ createGenerator し、seedは毎回上書き" にします。

    cudaError_t cuderr = cudaSetDevice(ctx->device_id);
    if (cuderr != cudaSuccess) {
        return DeviceError;
    }

    if (!ctx->inited) {
        // 初めて => create generator
        curandGenerator_t gen;
        curandStatus_t st = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        if (st != CURAND_STATUS_SUCCESS) {
            return DeviceError;
        }
        ctx->gen = gen;
        ctx->inited = true;
    }

    // 毎回 seed を上書き
    if (ctx->seed != seed) {
        curandStatus_t st = curandSetPseudoRandomGeneratorSeed(ctx->gen, seed);
        if (st != CURAND_STATUS_SUCCESS) {
            return DeviceError;
        }
        ctx->seed = seed;
    }

    return Success;
}


//==============================================================
// (1) NVIDIA 正規分布
//==============================================================
extern "C"
ZenuStatus zenu_compute_normal_distribution_nvidia(
    void* dst,
    int   num_elements,
    float mean,
    float stddev,
    ZenuDataType data_type,
    int device_id,
    unsigned long long seed
)
{
    if (!dst)                   return InvalidArgument;
    if (num_elements <= 0)      return InvalidArgument;
    if (stddev < 0.f)           return InvalidArgument;

    RandGPUContext* ctx = getRandGPUContext(device_id);
    if (!ctx) {
        return DeviceError; // no slot
    }
    ZenuStatus zst = ensureInitedGPUContext(ctx, seed);
    if (zst != Success) {
        return zst;
    }

    // generate
    curandStatus_t err;
    if (data_type == f32)
    {
        err = curandGenerateNormal(
            ctx->gen,
            static_cast<float*>(dst),
            (size_t)num_elements,
            mean,
            stddev
        );
    }
    else if (data_type == f64)
    {
        err = curandGenerateNormalDouble(
            ctx->gen,
            static_cast<double*>(dst),
            (size_t)num_elements,
            (double)mean,
            (double)stddev
        );
    }
    else {
        return InvalidArgument;
    }

    if (err != CURAND_STATUS_SUCCESS) {
        return DeviceError;
    }

    cudaError_t cuderr = cudaDeviceSynchronize();
    if (cuderr != cudaSuccess) {
        return DeviceError;
    }

    return Success;
}

//==============================================================
// (2) NVIDIA 一様分布
//==============================================================
__global__
static void scale_uniform_f32_kernel(float* arr, int n, float low, float range)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx<n) {
        float v = arr[idx];
        arr[idx] = v * range + low;
    }
}

__global__
static void scale_uniform_f64_kernel(double* arr, int n, double low, double range)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx<n) {
        double v = arr[idx];
        arr[idx] = v*range + low;
    }
}

extern "C"
ZenuStatus zenu_compute_uniform_distribution_nvidia(
    void* dst,
    int   num_elements,
    float low,
    float high,
    ZenuDataType data_type,
    int device_id,
    unsigned long long seed
)
{
    if (!dst)               return InvalidArgument;
    if (num_elements <= 0)  return InvalidArgument;
    if (low>high)           return InvalidArgument;

    RandGPUContext* ctx = getRandGPUContext(device_id);
    if (!ctx) {
        return DeviceError;
    }
    ZenuStatus zst = ensureInitedGPUContext(ctx, seed);
    if (zst != Success) {
        return zst;
    }

    // [0,1) generate
    curandStatus_t st;
    cudaError_t cuderr;
    if (data_type == f32)
    {
        st = curandGenerateUniform(
            ctx->gen,
            static_cast<float*>(dst),
            (size_t)num_elements
        );
        if (st != CURAND_STATUS_SUCCESS) {
            return DeviceError;
        }

        cuderr = cudaDeviceSynchronize();
        if (cuderr != cudaSuccess) {
            return DeviceError;
        }

        // scale
        float range = (float)(high - low);
        dim3 block(256);
        dim3 grid((num_elements + block.x -1)/block.x);
        scale_uniform_f32_kernel<<<grid, block>>>(
            static_cast<float*>(dst),
            num_elements,
            low,
            range
        );
        cuderr = cudaDeviceSynchronize();
        if (cuderr != cudaSuccess) {
            return DeviceError;
        }
    }
    else if (data_type == f64)
    {
        st = curandGenerateUniformDouble(
            ctx->gen,
            static_cast<double*>(dst),
            (size_t)num_elements
        );
        if (st != CURAND_STATUS_SUCCESS) {
            return DeviceError;
        }

        cuderr = cudaDeviceSynchronize();
        if (cuderr != cudaSuccess) {
            return DeviceError;
        }

        double range = (double)(high - low);
        dim3 block(256);
        dim3 grid((num_elements + block.x -1)/block.x);
        scale_uniform_f64_kernel<<<grid, block>>>(
            static_cast<double*>(dst),
            num_elements,
            (double)low,
            range
        );
        cuderr = cudaDeviceSynchronize();
        if (cuderr != cudaSuccess) {
            return DeviceError;
        }
    }
    else
    {
        return InvalidArgument;
    }

    return Success;
}


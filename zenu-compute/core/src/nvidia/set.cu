#include <cuda_runtime.h>
#include "zenu_compute_memory.h"
#include <cstdio>   // デバッグ目的などで使用

//------------------------------------------------------------------------------
// カーネル (単精度 float 用):
//   GPU上の value[0] を読み取り、dst 全体をその値で埋める
//------------------------------------------------------------------------------
__global__ void set_kernel_f32(float* dst, const float* src, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = src[0];  // GPU上の src[0] を読み取って書き込む
    }
}

//------------------------------------------------------------------------------
// カーネル (倍精度 double 用):
//   GPU上の value[0] を読み取り、dst 全体をその値で埋める
//------------------------------------------------------------------------------
__global__ void set_kernel_f64(double* dst, const double* src, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = src[0];  // GPU上の src[0] を読み取って書き込む
    }
}

//------------------------------------------------------------------------------
// 実装: zenu_compute_set_nvidia
//------------------------------------------------------------------------------
ZenuStatus zenu_compute_set_nvidia(void* dst, void* value, int num_bytes, ZenuDataType type)
{
    // 引数のチェック
    if (!dst || !value || (num_bytes <= 0)) {
        return InvalidArgument;
    }

    cudaError_t err;

    // ブロック・グリッドの設定 (簡易例)
    const int blockSize = 256;

    if (type == f32) {
        // f32 (float) の場合
        int N = num_bytes / static_cast<int>(sizeof(float));
        if (N <= 0) {
            return InvalidArgument; // 要素数が0以下の場合はエラー
        }

        int gridSize = (N + blockSize - 1) / blockSize;

        // カーネル起動
        set_kernel_f32<<<gridSize, blockSize>>>(
            static_cast<float*>(dst),
            static_cast<float*>(value),
            N
        );

        // カーネルエラーのチェック
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return DeviceError;
        }

        // カーネル完了待ち
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return DeviceError;
        }

    } else if (type == f64) {
        // f64 (double) の場合
        int N = num_bytes / static_cast<int>(sizeof(double));
        if (N <= 0) {
            return InvalidArgument;
        }

        int gridSize = (N + blockSize - 1) / blockSize;

        // カーネル起動
        set_kernel_f64<<<gridSize, blockSize>>>(
            static_cast<double*>(dst),
            static_cast<double*>(value),
            N
        );

        // カーネルエラーのチェック
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return DeviceError;
        }

        // カーネル完了待ち
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return DeviceError;
        }

    } else {
        // f32, f64 以外の型をサポートしない場合はエラー
        return InvalidArgument;
    }

    // 正常終了
    return Success;
}


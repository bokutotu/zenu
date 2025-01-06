#include "zenu_compute.h"
#include "utils.h"

//------------------------------------------------------------------------------
// 汎用カーネル: 配列 ptr[0..count-1] の全要素を val で初期化する
//------------------------------------------------------------------------------
template<typename T>
__global__ void SetKernel(T* ptr, T val, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ptr[idx] = val;
    }
}

//------------------------------------------------------------------------------
// GPUメモリ上のブロック（dst）を、value で埋める
//   - num_bytes は dst の総バイト数
//   - value は f32/f64 に応じたサイズのデータが格納されたポインタ
//------------------------------------------------------------------------------
extern "C"
ZenuStatus zenu_compute_set_nvidia(void* dst, void* value, int num_bytes, ZenuDataType type)
{
    // 入力チェック
    if (!dst || !value || num_bytes <= 0) {
        return InvalidArgument;
    }

    // 配列要素数 (float なら 4バイト刻み、double なら 8バイト刻み)
    int count = 0;
    cudaError_t err;

    if (type == f32) {
        count = num_bytes / sizeof(float);
        float val = *(float*)value;

        // ブロック/グリッド次元の設定
        dim3 block(256);
        dim3 grid((count + block.x - 1) / block.x);

        // カーネル起動
        SetKernel<float><<<grid, block>>>((float*)dst, val, count);

        // 同期 & エラーチェック
        err = cudaDeviceSynchronize();
        return convertCudaError(err);
    }
    else /* (type == f64) */ {
        count = num_bytes / sizeof(double);
        double val = *(double*)value;

        dim3 block(256);
        dim3 grid((count + block.x - 1) / block.x);

        SetKernel<double><<<grid, block>>>((double*)dst, val, count);

        err = cudaDeviceSynchronize();
        return convertCudaError(err);
    }
}

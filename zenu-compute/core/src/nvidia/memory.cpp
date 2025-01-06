#include "zenu_compute.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

ZenuStatus zenu_compute_malloc_nvidia(void** ptr, int num_bytes) {
    cudaError_t err = cudaMalloc(ptr, num_bytes);
    if (err != cudaSuccess) {
        return ZenuStatus::OutOfMemory;
    }
    return ZenuStatus::Success;
}

void zenu_compute_free_nvidia(void* ptr) {
    cudaFree(ptr);
}

//------------------------------------------------------------------------------
// CPU → GPU コピー
//   dst: GPUメモリ上のポインタ
//   src: CPUメモリ上のポインタ
//   num_bytes: コピーするバイト数
//------------------------------------------------------------------------------
ZenuStatus zenu_compute_cpu_to_nvidia(void* dst, void* src, int num_bytes)
{
    if (!dst || !src || (num_bytes <= 0)) {
        return InvalidArgument;
    }
    // cudaMemcpyHostToDevice でコピー
    cudaError_t err = cudaMemcpy(dst, src, num_bytes, cudaMemcpyHostToDevice);
    return convertCudaError(err);
}

//------------------------------------------------------------------------------
// GPU → CPU コピー
//   dst: CPUメモリ上のポインタ
//   src: GPUメモリ上のポインタ
//   num_bytes: コピーするバイト数
//------------------------------------------------------------------------------
ZenuStatus zenu_compute_nvidia_to_cpu(void* dst, void* src, int num_bytes)
{
    if (!dst || !src || (num_bytes <= 0)) {
        return InvalidArgument;
    }
    // cudaMemcpyDeviceToHost でコピー
    cudaError_t err = cudaMemcpy(dst, src, num_bytes, cudaMemcpyDeviceToHost);
    return convertCudaError(err);
}

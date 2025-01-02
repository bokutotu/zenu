#include "zenu_compute.h"

#include <cuda_runtime.h>

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

void zenu_compute_copy_nvidia(void* dst, void* src, int num_bytes) {
    cudaMemcpy(dst, src, num_bytes, cudaMemcpyDefault);
}

void zenu_compute_set_nvidia(void* dst, int value, int num_bytes) {
    cudaMemset(dst, value, num_bytes);
}

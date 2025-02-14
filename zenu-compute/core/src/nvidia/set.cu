#include <cuda_runtime.h>
#include "zenu_compute_memory.h"
#include <cstdio>

__global__ void set_kernel_f32(float* dst, const float* src, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = src[0];
    }
}

__global__ void set_kernel_f64(double* dst, const double* src, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = src[0];
    }
}

ZenuStatus zenu_compute_set_nvidia(void* dst, void* value, int num_bytes, ZenuDataType type)
{
    if (!dst || !value || (num_bytes <= 0)) {
        return InvalidArgument;
    }

    cudaError_t err;

    const int blockSize = 256;

    if (type == f32) {
        int N = num_bytes / static_cast<int>(sizeof(float));
        if (N <= 0) {
            return InvalidArgument;
        }

        int gridSize = (N + blockSize - 1) / blockSize;

        set_kernel_f32<<<gridSize, blockSize>>>(
            static_cast<float*>(dst),
            static_cast<float*>(value),
            N
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return DeviceError;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return DeviceError;
        }

    } else if (type == f64) {
        int N = num_bytes / static_cast<int>(sizeof(double));
        if (N <= 0) {
            return InvalidArgument;
        }

        int gridSize = (N + blockSize - 1) / blockSize;

        set_kernel_f64<<<gridSize, blockSize>>>(
            static_cast<double*>(dst),
            static_cast<double*>(value),
            N
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return DeviceError;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return DeviceError;
        }

    } else {
        return InvalidArgument;
    }

    return Success;
}


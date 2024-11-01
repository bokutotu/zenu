#include "element_wise.h"

#include <limits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename T>
__device__ void warp_reduce(volatile T* sdata, volatile int* sindex, unsigned int tid) {
    if (BLOCK_SIZE >= 64) {
        if (sdata[tid] < sdata[tid + 32]) {
            sdata[tid] = sdata[tid + 32];
            sindex[tid] = sindex[tid + 32];
        }
    }
    if (BLOCK_SIZE >= 32) {
        if (sdata[tid] < sdata[tid + 16]) {
            sdata[tid] = sdata[tid + 16];
            sindex[tid] = sindex[tid + 16];
        }
    }
    if (BLOCK_SIZE >= 16) {
        if (sdata[tid] < sdata[tid + 8]) {
            sdata[tid] = sdata[tid + 8];
            sindex[tid] = sindex[tid + 8];
        }
    }
    if (BLOCK_SIZE >= 8) {
        if (sdata[tid] < sdata[tid + 4]) {
            sdata[tid] = sdata[tid + 4];
            sindex[tid] = sindex[tid + 4];
        }
    }
    if (BLOCK_SIZE >= 4) {
        if (sdata[tid] < sdata[tid + 2]) {
            sdata[tid] = sdata[tid + 2];
            sindex[tid] = sindex[tid + 2];
        }
    }
    if (BLOCK_SIZE >= 2) {
        if (sdata[tid] < sdata[tid + 1]) {
            sdata[tid] = sdata[tid + 1];
            sindex[tid] = sindex[tid + 1];
        }
    }
}

template <typename T>
__global__ void find_max_index(T* data, int* result, int size, int stride) {
    __shared__ T sdata[BLOCK_SIZE];
    __shared__ int sindex[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int grid_size = blockDim.x * 2 * gridDim.x;

    sdata[tid] = -std::numeric_limits<T>::infinity();
    sindex[tid] = -1;

    while (i < size) {
        if (data[i * stride] > sdata[tid]) {
            sdata[tid] = data[i * stride];
            sindex[tid] = i;
        }
        if (i + blockDim.x < size && data[(i + blockDim.x) * stride] > sdata[tid]) {
            sdata[tid] = data[(i + blockDim.x) * stride];
            sindex[tid] = i + blockDim.x;
        }
        i += grid_size;
    }
    __syncthreads();

    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            if (sdata[tid] < sdata[tid + 256]) {
                sdata[tid] = sdata[tid + 256];
                sindex[tid] = sindex[tid + 256];
            }
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            if (sdata[tid] < sdata[tid + 128]) {
                sdata[tid] = sdata[tid + 128];
                sindex[tid] = sindex[tid + 128];
            }
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            if (sdata[tid] < sdata[tid + 64]) {
                sdata[tid] = sdata[tid + 64];
                sindex[tid] = sindex[tid + 64];
            }
        }
        __syncthreads();
    }

    if (tid < 32) warp_reduce<T>(sdata, sindex, tid);

    if (tid == 0) atomicMax(result, sindex[0]);
}

template <typename T>
void launch_find_max_index(T* d_data, int* d_result, int size, int stride) {
    int block_size = BLOCK_SIZE;
    int grid_size = (size + block_size - 1) / block_size;
    find_max_index<T><<<grid_size, block_size>>>(d_data, d_result, size, stride);
}

template void launch_find_max_index<float>(float* d_data, int* d_result, int size, int stride);
template void launch_find_max_index<double>(double* d_data, int* d_result, int size, int stride);

void array_max_idx_float(float *a, int size, int stride, int *out) {
    float* d_data;
    int* d_result;

    cudaMalloc((void**)&d_data, size * stride * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(int));

    cudaMemcpy(d_data, a, size * stride * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));

    launch_find_max_index(d_data, d_result, size, stride);

    cudaMemcpy(out, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_result);
}

void array_max_idx_double(double *a, int size, int stride, int *out) {
    double* d_data;
    int* d_result;

    cudaMalloc((void**)&d_data, size * stride * sizeof(double));
    cudaMalloc((void**)&d_result, sizeof(int));

    cudaMemcpy(d_data, a, size * stride * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));

    launch_find_max_index(d_data, d_result, size, stride);

    cudaMemcpy(out, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_result);
}

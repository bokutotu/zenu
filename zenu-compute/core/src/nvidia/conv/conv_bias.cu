#include "zenu_compute_type.h"
#include "zenu_compute_conv.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

template<typename T>
__global__ void conv_forward_bias_kernel(const T* input, const T* bias, T* output,
                                           size_t N, size_t C, size_t H, size_t W)
{
    size_t total = N * C * H * W;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < total; idx += blockDim.x * gridDim.x) {
        size_t c = (idx / (W * H)) % C;
        output[idx] = input[idx] + bias[c];
    }
}

template<typename T>
__global__ void compute_bias_grad_partial(const T* d_output, T* partial,
                                            size_t N, size_t C, size_t H, size_t W)
{
    size_t channel = blockIdx.y;
    size_t spatial = H * W;
    size_t S = N * spatial;
    size_t tid = threadIdx.x;
    T sum = 0;

    for (size_t s = blockIdx.x * blockDim.x + tid; s < S; s += blockDim.x * gridDim.x) {
        size_t n = s / spatial;
        size_t i = s % spatial;
        size_t index = n * (C * spatial) + channel * spatial + i;
        sum += d_output[index];
    }

    extern __shared__ unsigned char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[channel * gridDim.x + blockIdx.x] = sdata[0];
    }
}

template<typename T>
__global__ void finalize_bias_grad(const T* partial, T* d_bias, size_t numBlocks, size_t C)
{
    size_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel < C) {
        T sum = 0;
        for (size_t i = 0; i < numBlocks; i++) {
            sum += partial[channel * numBlocks + i];
        }
        d_bias[channel] = sum;
    }
}

ZenuStatus zenu_compute_conv_forward_bias_nvidia(
    size_t* input_shape,
    size_t* bias_shape,
    size_t conv_dim,
    ZenuDataType data_type,
    const void* input,
    const void* bias,
    void* output)
{
    if (!input_shape || !bias_shape || !input || !bias || !output) {
        return InvalidArgument;
    }
    if (data_type != ZenuDataType::f32 && data_type != ZenuDataType::f64) {
        return InvalidArgument;
    }

    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H, W;
    if (conv_dim == 1) {
        H = 1;
        W = input_shape[2];
    } else if (conv_dim == 2) {
        H = input_shape[2];
        W = input_shape[3];
    } else {
        return InvalidArgument;
    }

    size_t total = N * C * H * W;
    const int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    if (data_type == ZenuDataType::f32) {
        conv_forward_bias_kernel<float><<<gridSize, blockSize>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(bias),
            static_cast<float*>(output),
            N, C, H, W);
    }
    else {
        conv_forward_bias_kernel<double><<<gridSize, blockSize>>>(
            static_cast<const double*>(input),
            static_cast<const double*>(bias),
            static_cast<double*>(output),
            N, C, H, W);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in conv_forward_bias_kernel: %s\n",
                cudaGetErrorString(err));
        return CudnnError;
    }
    return Success;
}

ZenuStatus zenu_compute_conv_bkwd_bias_nvidia(
    size_t* input_shape,
    size_t* bias_shape,
    size_t conv_dim,
    ZenuDataType data_type,
    const void* d_output,
    void* d_bias,
    void* workspace)
{
    if (!input_shape || !bias_shape || !d_output || !d_bias || !workspace) {
        return InvalidArgument;
    }
    if (data_type != ZenuDataType::f32 && data_type != ZenuDataType::f64) {
        return InvalidArgument;
    }

    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H, W;
    if (conv_dim == 1) {
        H = 1;
        W = input_shape[2];
    } else if (conv_dim == 2) {
        H = input_shape[2];
        W = input_shape[3];
    } else {
        return InvalidArgument;
    }

    size_t spatial = H * W;
    size_t S = N * spatial;
    int blockSize = 256;
    int gridX = (S + blockSize - 1) / blockSize;
    gridX = std::min(gridX, 256);
    dim3 grid(gridX, C);

    if (data_type == ZenuDataType::f32) {
        float* d_partial = static_cast<float*>(workspace);
        compute_bias_grad_partial<float><<<grid, blockSize, blockSize * sizeof(float)>>>(
            static_cast<const float*>(d_output),
            d_partial,
            N, C, H, W);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in compute_bias_grad_partial (f32): %s\n",
                    cudaGetErrorString(err));
            return CudnnError;
        }
        int threadsFinal = 256;
        int blocksFinal = (C + threadsFinal - 1) / threadsFinal;
        finalize_bias_grad<float><<<blocksFinal, threadsFinal>>>(
            d_partial,
            static_cast<float*>(d_bias),
            gridX,
            C);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in finalize_bias_grad (f32): %s\n",
                    cudaGetErrorString(err));
            return CudnnError;
        }
    }
    else {
        double* d_partial = static_cast<double*>(workspace);
        compute_bias_grad_partial<double><<<grid, blockSize, blockSize * sizeof(double)>>>(
            static_cast<const double*>(d_output),
            d_partial,
            N, C, H, W);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in compute_bias_grad_partial (f64): %s\n",
                    cudaGetErrorString(err));
            return CudnnError;
        }
        int threadsFinal = 256;
        int blocksFinal = (C + threadsFinal - 1) / threadsFinal;
        finalize_bias_grad<double><<<blocksFinal, threadsFinal>>>(
            d_partial,
            static_cast<double*>(d_bias),
            gridX,
            C);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error in finalize_bias_grad (f64): %s\n",
                    cudaGetErrorString(err));
            return CudnnError;
        }
    }
    return Success;
}

ZenuStatus zenu_compute_bkwd_bias_get_workspace_nvidia(
    size_t* input_shape,
    size_t conv_dim,
    ZenuDataType data_type,
    size_t* workspace_size)
{
    if (!input_shape || !workspace_size) {
        return InvalidArgument;
    }
    if (data_type != ZenuDataType::f32 && data_type != ZenuDataType::f64) {
        return InvalidArgument;
    }
    
    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H, W;
    if (conv_dim == 1) {
        H = 1;
        W = input_shape[2];
    } else if (conv_dim == 2) {
        H = input_shape[2];
        W = input_shape[3];
    } else {
        return InvalidArgument;
    }
    
    size_t spatial = H * W;
    size_t S = N * spatial;
    int blockSize = 256;
    int gridX = (S + blockSize - 1) / blockSize;
    gridX = std::min(gridX, 256);
    
    size_t num_elements = C * gridX;  // partial 配列の要素数
    if (data_type == ZenuDataType::f32) {
        *workspace_size = num_elements * sizeof(float);
    }
    else { // f64
        *workspace_size = num_elements * sizeof(double);
    }
    
    return Success;
}


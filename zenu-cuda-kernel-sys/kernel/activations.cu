#include "activations.h"

template <typename T>
__global__ void relu_kernel(T *input , T* output, T alpha, int size, int input_stride, int output_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx * output_stride] = input[idx * input_stride] > 0 ? input[idx * input_stride] : alpha * input[idx * input_stride];
    }
}

template <typename T>
__global__ void relu_background_mask(T *input, T *mask, T alpha, int size, int input_stride, int mask_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mask[idx * mask_stride] = input[idx * input_stride] > 0 ? 1 : alpha * -1;
    }
}

void relu_float(float *input , float* output, float alpha, int size, int input_stride, int output_stride) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    relu_kernel<float><<<grid_size, block_size>>>(input, output, alpha, size, input_stride, output_stride);
}

void relu_double(double *input , double* output, double alpha, int size, int input_stride, int output_stride) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    relu_kernel<double><<<grid_size, block_size>>>(input, output, alpha, size, input_stride, output_stride);
}

void relu_backward_mask_float(float *input, float *mask, float alpha, int size, int input_stride, int mask_stride) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    relu_background_mask<float><<<grid_size, block_size>>>(input, mask, alpha, size, input_stride, mask_stride);
}

void relu_backward_mask_double(double *input, double *mask, double alpha, int size, int input_stride, int mask_stride) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    relu_background_mask<double><<<grid_size, block_size>>>(input, mask, alpha, size, input_stride, mask_stride);
}
// #include "activations.h"
// #include <cuda_runtime.h>
//
// template <typename T, bool UseStride>
// __global__ void relu_kernel_optimized(const T* __restrict__ input, T* __restrict__ output, T alpha, int size, int input_stride, int output_stride) {
//     int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
//     
//     T in[4], out[4];
//
//     #pragma unroll
//     for (int i = 0; i < 4; ++i) {
//         if (idx + i * blockDim.x < size) {
//             in[i] = UseStride ? input[(idx + i * blockDim.x) * input_stride] : input[idx + i * blockDim.x];
//             out[i] = in[i] > 0 ? in[i] : alpha * in[i];
//             if (UseStride) {
//                 output[(idx + i * blockDim.x) * output_stride] = out[i];
//             } else {
//                 output[idx + i * blockDim.x] = out[i];
//             }
//         }
//     }
// }
//
// template <typename T, bool UseStride>
// __global__ void relu_backward_mask_optimized(const T* __restrict__ input, T* __restrict__ mask, T alpha, int size, int input_stride, int mask_stride) {
//     int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
//     
//     T in[4], m[4];
//
//     #pragma unroll
//     for (int i = 0; i < 4; ++i) {
//         if (idx + i * blockDim.x < size) {
//             in[i] = UseStride ? input[(idx + i * blockDim.x) * input_stride] : input[idx + i * blockDim.x];
//             m[i] = in[i] > 0 ? 1 : alpha * -1;
//             if (UseStride) {
//                 mask[(idx + i * blockDim.x) * mask_stride] = m[i];
//             } else {
//                 mask[idx + i * blockDim.x] = m[i];
//             }
//         }
//     }
// }
//
// template <typename T>
// void relu_optimized(T* input, T* output, T alpha, int size, int input_stride, int output_stride) {
//     int block_size = 256;
//     int grid_size = (size + block_size * 4 - 1) / (block_size * 4);
//
//     if (input_stride == 1 && output_stride == 1) {
//         relu_kernel_optimized<T, false><<<grid_size, block_size>>>(input, output, alpha, size, input_stride, output_stride);
//     } else {
//         relu_kernel_optimized<T, true><<<grid_size, block_size>>>(input, output, alpha, size, input_stride, output_stride);
//     }
// }
//
// template <typename T>
// void relu_backward_mask_optimized(T* input, T* mask, T alpha, int size, int input_stride, int mask_stride) {
//     int block_size = 256;
//     int grid_size = (size + block_size * 4 - 1) / (block_size * 4);
//
//     if (input_stride == 1 && mask_stride == 1) {
//         relu_backward_mask_optimized<T, false><<<grid_size, block_size>>>(input, mask, alpha, size, input_stride, mask_stride);
//     } else {
//         relu_backward_mask_optimized<T, true><<<grid_size, block_size>>>(input, mask, alpha, size, input_stride, mask_stride);
//     }
// }
//
// extern "C" {
//     void relu_float(float* input, float* output, float alpha, int size, int input_stride, int output_stride) {
//         relu_optimized<float>(input, output, alpha, size, input_stride, output_stride);
//     }
//
//     void relu_double(double* input, double* output, double alpha, int size, int input_stride, int output_stride) {
//         relu_optimized<double>(input, output, alpha, size, input_stride, output_stride);
//     }
//
//     void relu_backward_mask_float(float* input, float* mask, float alpha, int size, int input_stride, int mask_stride) {
//         relu_backward_mask_optimized<float>(input, mask, alpha, size, input_stride, mask_stride);
//     }
//
//     void relu_backward_mask_double(double* input, double* mask, double alpha, int size, int input_stride, int mask_stride) {
//         relu_backward_mask_optimized<double>(input, mask, alpha, size, input_stride, mask_stride);
//     }
// }

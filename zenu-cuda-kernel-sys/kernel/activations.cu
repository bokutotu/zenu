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
    int block_size = 128;
    int grid_size = (size + block_size - 1) / block_size;
    relu_kernel<float><<<grid_size, block_size>>>(input, output, alpha, size, input_stride, output_stride);
}

void relu_double(double *input , double* output, double alpha, int size, int input_stride, int output_stride) {
    int block_size = 128;
    int grid_size = (size + block_size - 1) / block_size;
    relu_kernel<double><<<grid_size, block_size>>>(input, output, alpha, size, input_stride, output_stride);
}

void relu_backward_mask_float(float *input, float *mask, float alpha, int size, int input_stride, int mask_stride) {
    int block_size = 128;
    int grid_size = (size + block_size - 1) / block_size;
    relu_background_mask<float><<<grid_size, block_size>>>(input, mask, alpha, size, input_stride, mask_stride);
}

void relu_backward_mask_double(double *input, double *mask, double alpha, int size, int input_stride, int mask_stride) {
    int block_size = 128;
    int grid_size = (size + block_size - 1) / block_size;
    relu_background_mask<double><<<grid_size, block_size>>>(input, mask, alpha, size, input_stride, mask_stride);
}

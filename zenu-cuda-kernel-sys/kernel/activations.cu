#include "activations.h"

template <typename T>
__global__ void relu_kernel(T *input , T* output, T alpha, int size, int input_stride, int output_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx * output_stride] = input[idx * input_stride] > 0 ? input[idx * input_stride] : alpha * input[idx * input_stride];
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

#include "array_array.h"
#include "cuda_runtime.h"

#define CUDA_VEC_VEC_OP(op, func, assign_func, type)                                                                                          \
__global__ void vector_vector_##func##_##type(type* vec1, int stride1, type* vec2, int stride2, int size, type* result, int stride_result) {  \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                          \
    if (idx < size) {                                                                                                                         \
        result[idx * stride_result] = vec1[idx * stride1] op vec2[idx * stride2];                                                             \
    }                                                                                                                                         \
}                                                                                                                                             \
__global__ void vector_vector_##assign_func##_##type(type* vec1, int stride1, type* vec2, int stride2, int size) {                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                                          \
    if (idx < size) {                                                                                                                         \
        vec1[idx * stride1] op## = vec2[idx * stride2];                                                                                       \
    }                                                                                                                                         \
}

CUDA_VEC_VEC_OP(+, add, add_assign, float)
CUDA_VEC_VEC_OP(-, sub, sub_assign, float)
CUDA_VEC_VEC_OP(*, mul, mul_assign, float)
CUDA_VEC_VEC_OP(/, div, div_assign, float)
CUDA_VEC_VEC_OP(+, add, add_assign, double)
CUDA_VEC_VEC_OP(-, sub, sub_assign, double)
CUDA_VEC_VEC_OP(*, mul, mul_assign, double)
CUDA_VEC_VEC_OP(/, div, div_assign, double)

const int BLOCK_SIZE = 256;

#define DEFINE_ARRAY_ARRAY_WRAPPER(type, op)                                                                                   \
void array_array_##op##_##type(type *a, int stride_a, type *b, int stride_b, type *out, int stride_out, int size) {            \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                                                       \
    vector_vector_##op##_##type<<<gridSize, BLOCK_SIZE>>>(a, stride_a, b, stride_b, size, out, stride_out);                    \
}                                                                                                                              \
void array_array_##op##_assign_##type(type *a, int stride_a, type *b, int stride_b, int size) {                                \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                                                       \
    vector_vector_##op##_assign_##type<<<gridSize, BLOCK_SIZE>>>(a, stride_a, b, stride_b, size);                              \
}

DEFINE_ARRAY_ARRAY_WRAPPER(float, add)
DEFINE_ARRAY_ARRAY_WRAPPER(float, sub)
DEFINE_ARRAY_ARRAY_WRAPPER(float, mul)
DEFINE_ARRAY_ARRAY_WRAPPER(float, div)

DEFINE_ARRAY_ARRAY_WRAPPER(double, add)
DEFINE_ARRAY_ARRAY_WRAPPER(double, sub)
DEFINE_ARRAY_ARRAY_WRAPPER(double, mul)
DEFINE_ARRAY_ARRAY_WRAPPER(double, div)

template <typename T>
__global__ void conv_bias_add_kernel(const T *input, T *output, int channel_stride, 
                                     const T *bias, int bias_size, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        int channel = (idx / channel_stride) % bias_size;
        output[idx] = input[idx] + bias[channel];
    }
}

void conv_bias_add_float(const float *input, float *output, int channel_stride, const float *bias, int bias_size, int total_elements) {
    dim3 block_size(256);
    dim3 grid_size((total_elements + block_size.x - 1) / block_size.x);

    conv_bias_add_kernel<<<grid_size, block_size>>>(input, output, channel_stride, bias, bias_size, total_elements);
    cudaDeviceSynchronize();
}

void conv_bias_add_double(const double *input, double *output, int channel_stride, const double *bias, int bias_size, int total_elements) {
    dim3 block_size(256);
    dim3 grid_size((total_elements + block_size.x - 1) / block_size.x);

    conv_bias_add_kernel<<<grid_size, block_size>>>(input, output, channel_stride, bias, bias_size, total_elements);
    cudaDeviceSynchronize();
}

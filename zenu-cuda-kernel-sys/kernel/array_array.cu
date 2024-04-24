#include "array_array.h"

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

#define DEFINE_ARRAY_ARRAY_WRAPPER(type, op)                                                                        \
void array_array_##op##_##type(type *a, int stride_a, type *b, int stride_b, type *out, int stride_out, int size) { \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                                            \
    vector_vector_##op##_##type<<<gridSize, BLOCK_SIZE>>>(a, stride_a, b, stride_b, size, out, stride_out);         \
}                                                                                                                   \
void array_array_##op##_assign_##type(type *a, int stride_a, type *b, int stride_b, int size) {                     \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                                            \
    vector_vector_##op##_assign_##type<<<gridSize, BLOCK_SIZE>>>(a, stride_a, b, stride_b, size);                   \
}

DEFINE_ARRAY_ARRAY_WRAPPER(float, add)
DEFINE_ARRAY_ARRAY_WRAPPER(double, add)
DEFINE_ARRAY_ARRAY_WRAPPER(float, sub)
DEFINE_ARRAY_ARRAY_WRAPPER(double, sub)
DEFINE_ARRAY_ARRAY_WRAPPER(float, mul)
DEFINE_ARRAY_ARRAY_WRAPPER(double, mul)
DEFINE_ARRAY_ARRAY_WRAPPER(float, div)
DEFINE_ARRAY_ARRAY_WRAPPER(double, div)

#include "array_scalar.h"

#define CUDA_VEC_SCALAR_OP(op, func, assign_func, type)                                                     \
__global__ void vector_scalar_##func##_##type(type* vec, int size, int stride, type scalar, type* result) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                        \
    if (idx < size) {                                                                                       \
        result[idx * stride] = vec[idx * stride] op scalar;                                                 \
    }                                                                                                       \
}                                                                                                           \
__global__ void vector_scalar_##assign_func##_##type(type* vec, int size, int stride, type scalar) {        \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                        \
    if (idx < size) {                                                                                       \
        vec[idx * stride] op## = scalar;                                                                    \
    }                                                                                                       \
}

CUDA_VEC_SCALAR_OP(+, add, add_assign, float)
CUDA_VEC_SCALAR_OP(-, sub, sub_assign, float)
CUDA_VEC_SCALAR_OP(*, mul, mul_assign, float)
CUDA_VEC_SCALAR_OP(/, div, div_assign, float)
CUDA_VEC_SCALAR_OP(+, add, add_assign, double)
CUDA_VEC_SCALAR_OP(-, sub, sub_assign, double)
CUDA_VEC_SCALAR_OP(*, mul, mul_assign, double)
CUDA_VEC_SCALAR_OP(/, div, div_assign, double)

const int BLOCK_SIZE = 256;

#define DEFINE_ARRAY_SCALAR_WRAPPER(type, op)                                                       \
void array_scalar_##op##_##type(type *a, int size, int stride, type scalar, type *out) {            \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                            \
    vector_scalar_##op##_##type<<<gridSize, BLOCK_SIZE>>>(a, size, stride, scalar, out);            \
}                                                                                                   \
void array_scalar_##op##_assign_##type(type *a, int size, int stride, type scalar) {                \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                            \
    vector_scalar_##op##_assign_##type<<<gridSize, BLOCK_SIZE>>>(a, size, stride, scalar);          \
}

DEFINE_ARRAY_SCALAR_WRAPPER(float, add)
DEFINE_ARRAY_SCALAR_WRAPPER(double, add)
DEFINE_ARRAY_SCALAR_WRAPPER(float, sub)
DEFINE_ARRAY_SCALAR_WRAPPER(double, sub)
DEFINE_ARRAY_SCALAR_WRAPPER(float, mul)
DEFINE_ARRAY_SCALAR_WRAPPER(double, mul)
DEFINE_ARRAY_SCALAR_WRAPPER(float, div)
DEFINE_ARRAY_SCALAR_WRAPPER(double, div)

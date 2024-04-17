#include "array_scalar.h"
#include "cuda_runtime.h"

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

#define CUDA_VEC_FUNC(func, type)                                                          \
__global__ void vector_##func##_##type(type* vec, int size, int stride, type* result) {    \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                       \
    if (idx < size) {                                                                      \
        result[idx * stride] = func(vec[idx * stride]);                                    \
    }                                                                                      \
}                                                                                          \
__global__ void vector_##func##_assign_##type(type* vec, int size, int stride) {           \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                       \
    if (idx < size) {                                                                      \
        vec[idx * stride] = func(vec[idx * stride]);                                       \
    }                                                                                      \
}

CUDA_VEC_FUNC(sin, float)
CUDA_VEC_FUNC(cos, float)
CUDA_VEC_FUNC(tan, float)
CUDA_VEC_FUNC(asin, float)
CUDA_VEC_FUNC(acos, float)
CUDA_VEC_FUNC(atan, float)
CUDA_VEC_FUNC(sinh, float)
CUDA_VEC_FUNC(cosh, float)
CUDA_VEC_FUNC(tanh, float)
CUDA_VEC_FUNC(abs, float)
CUDA_VEC_FUNC(sqrt, float)
CUDA_VEC_FUNC(sin, double)
CUDA_VEC_FUNC(cos, double)
CUDA_VEC_FUNC(tan, double)
CUDA_VEC_FUNC(asin, double)
CUDA_VEC_FUNC(acos, double)
CUDA_VEC_FUNC(atan, double)
CUDA_VEC_FUNC(sinh, double)
CUDA_VEC_FUNC(cosh, double)
CUDA_VEC_FUNC(tanh, double)
CUDA_VEC_FUNC(abs, double)
CUDA_VEC_FUNC(sqrt, double)
CUDA_VEC_FUNC(exp, float)
CUDA_VEC_FUNC(exp, double)

#define DEFINE_ARRAY_FUNC_WRAPPER(type, func)                                                  \
void array_##func##_##type(type *a, int size, int stride, type *out) {                         \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                       \
    vector_##func##_##type<<<gridSize, BLOCK_SIZE>>>(a, size, stride, out);                    \
}                                                                                              \
void array_##func##_assign_##type(type *a, int size, int stride) {                             \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                       \
    vector_##func##_assign_##type<<<gridSize, BLOCK_SIZE>>>(a, size, stride);                  \
}

DEFINE_ARRAY_FUNC_WRAPPER(float, sin)
DEFINE_ARRAY_FUNC_WRAPPER(double, sin)
DEFINE_ARRAY_FUNC_WRAPPER(float, cos)
DEFINE_ARRAY_FUNC_WRAPPER(double, cos)
DEFINE_ARRAY_FUNC_WRAPPER(float, tan)
DEFINE_ARRAY_FUNC_WRAPPER(double, tan)
DEFINE_ARRAY_FUNC_WRAPPER(float, asin)
DEFINE_ARRAY_FUNC_WRAPPER(double, asin)
DEFINE_ARRAY_FUNC_WRAPPER(float, acos)
DEFINE_ARRAY_FUNC_WRAPPER(double, acos)
DEFINE_ARRAY_FUNC_WRAPPER(float, atan)
DEFINE_ARRAY_FUNC_WRAPPER(double, atan)
DEFINE_ARRAY_FUNC_WRAPPER(float, sinh)
DEFINE_ARRAY_FUNC_WRAPPER(double, sinh)
DEFINE_ARRAY_FUNC_WRAPPER(float, cosh)
DEFINE_ARRAY_FUNC_WRAPPER(double, cosh)
DEFINE_ARRAY_FUNC_WRAPPER(float, tanh)
DEFINE_ARRAY_FUNC_WRAPPER(double, tanh)
DEFINE_ARRAY_FUNC_WRAPPER(float, abs)
DEFINE_ARRAY_FUNC_WRAPPER(double, abs)
DEFINE_ARRAY_FUNC_WRAPPER(float, sqrt)
DEFINE_ARRAY_FUNC_WRAPPER(double, sqrt)
DEFINE_ARRAY_FUNC_WRAPPER(float, exp)
DEFINE_ARRAY_FUNC_WRAPPER(double, exp)

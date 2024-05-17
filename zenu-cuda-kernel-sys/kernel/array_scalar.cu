#include "array_scalar.h"
#include "cuda_runtime.h"

__global__ void clip_float(float* d_input, float* d_output, int size, int stride_in, int stride_out, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int idx_in = idx * stride_in;
        int idx_out = idx * stride_out;
        d_output[idx_in] = max(min(d_input[idx_out], max_val), min_val);
    }
}

__global__ void clip_double(double* d_input, double* d_output, int size, int stride_in, int stride_out, double min_val, double max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int idx_in = idx * stride_in;
        int idx_out = idx * stride_out;
        d_output[idx_in] = max(min(d_input[idx_out], max_val), min_val);
    }
}

__global__ void clip_float_assign(float* d_input, int size, int stride_in, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int idx_in = idx * stride_in;
        d_input[idx_in] = max(min(d_input[idx_in], max_val), min_val);
    }
}

__global__ void clip_double_assign(double* d_input, int size, int stride_in, double min_val, double max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int idx_in = idx * stride_in;
        d_input[idx_in] = max(min(d_input[idx_in], max_val), min_val);
    }
}

__global__ void clip_backward_float(float* input, float* mask, float max, float min, int size, int stride_in, int stride_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mask[idx * stride_mask] = (input[idx * stride_in] >= min && input[idx * stride_in] <= max) ? 1 : 0;
    }
}

__global__ void clip_backward_double(double* input, double* mask, double max, double min, int size, int stride_in, int stride_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mask[idx * stride_mask] = (input[idx * stride_in] >= min && input[idx * stride_in] <= max) ? 1 : 0;
    }
}

__global__ void clip_backward_assign_float(float* mask, float max, float min, int size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mask[idx * stride] = (mask[idx * stride] >= min && mask[idx * stride] <= max) ? 1 : 0;
    }
}

__global__ void clip_backward_assign_double(double* mask, double max, double min, int size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mask[idx * stride] = (mask[idx * stride] >= min && mask[idx * stride] <= max) ? 1 : 0;
    }
}

#define CUDA_VEC_SCALAR_OP(op, func, assign_func, type)                                                                     \
__global__ void vector_scalar_##func##_##type(type* vec, int size, int stride_v, type scalar, type* result, int stride_r) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                        \
    if (idx < size) {                                                                                                       \
        result[idx * stride_r] = vec[idx * stride_v] op scalar;                                                             \
    }                                                                                                                       \
}                                                                                                                           \
__global__ void vector_scalar_##assign_func##_##type(type* vec, int size, int stride, type scalar) {                        \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                                        \
    if (idx < size) {                                                                                                       \
        vec[idx * stride] op## = scalar;                                                                                    \
    }                                                                                                                       \
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

#define DEFINE_ARRAY_SCALAR_WRAPPER(type, op)                                                             \
void array_scalar_##op##_##type(type *a, int size, int stride_a, type scalar, type *out, int stride_o) {  \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                                  \
    vector_scalar_##op##_##type<<<gridSize, BLOCK_SIZE>>>(a, size, stride_a, scalar, out, stride_o);      \
}                                                                                                         \
void array_scalar_##op##_assign_##type(type *a, int size, int stride, type scalar) {                      \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                                  \
    vector_scalar_##op##_assign_##type<<<gridSize, BLOCK_SIZE>>>(a, size, stride, scalar);                \
}

DEFINE_ARRAY_SCALAR_WRAPPER(float, add)
DEFINE_ARRAY_SCALAR_WRAPPER(double, add)
DEFINE_ARRAY_SCALAR_WRAPPER(float, sub)
DEFINE_ARRAY_SCALAR_WRAPPER(double, sub)
DEFINE_ARRAY_SCALAR_WRAPPER(float, mul)
DEFINE_ARRAY_SCALAR_WRAPPER(double, mul)
DEFINE_ARRAY_SCALAR_WRAPPER(float, div)
DEFINE_ARRAY_SCALAR_WRAPPER(double, div)

void array_clip_float(float *input, float* output, int size, int stride_in, int stride_out, float min_val, float max_val) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_float<<<gridSize, BLOCK_SIZE>>>(input, output, size, stride_in, stride_out, min_val, max_val);
}

void array_clip_double(double *input, double* output, int size, int stride_in, int stride_out, double min_val, double max_val) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_double<<<gridSize, BLOCK_SIZE>>>(input, output, size, stride_in, stride_out, min_val, max_val);
}

void array_clip_assign_float(float *input, int size, int stride_in, float min_val, float max_val) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_float_assign<<<gridSize, BLOCK_SIZE>>>(input, size, stride_in, min_val, max_val);
}

void array_clip_assign_double(double *input, int size, int stride_in, double min_val, double max_val) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_double_assign<<<gridSize, BLOCK_SIZE>>>(input, size, stride_in, min_val, max_val);
}

void array_clip_backward_float(float *input, float *mask, float max, float min, int size, int stride_in, int stride_mask) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_backward_float<<<gridSize, BLOCK_SIZE>>>(input, mask, max, min, size, stride_in, stride_mask);
}

void array_clip_backward_double(double *input, double *mask, double max, double min, int size, int stride_in, int stride_mask) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_backward_double<<<gridSize, BLOCK_SIZE>>>(input, mask, max, min, size, stride_in, stride_mask);
}

void array_clip_backward_assign_float(float *mask, float max, float min, int size, int stride) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_backward_assign_float<<<gridSize, BLOCK_SIZE>>>(mask, max, min, size, stride);
}

void array_clip_backward_assign_double(double *mask, double max, double min, int size, int stride) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_backward_assign_double<<<gridSize, BLOCK_SIZE>>>(mask, max, min, size, stride);
}

#define CUDA_VEC_FUNC(func, type)                                                                                  \
__global__ void vector_##func##_##type(type* vec, int size, int stride_in, type* result, int stride_out) {         \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                               \
    if (idx < size) {                                                                                              \
        result[idx * stride_out] = func(vec[idx * stride_in]);                                                     \
    }                                                                                                              \
}                                                                                                                  \
__global__ void vector_##func##_assign_##type(type* vec, int size, int stride) {                                   \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                                               \
    if (idx < size) {                                                                                              \
        vec[idx * stride] = func(vec[idx * stride]);                                                               \
    }                                                                                                              \
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
CUDA_VEC_FUNC(exp, float)
CUDA_VEC_FUNC(log, float)

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
CUDA_VEC_FUNC(exp, double)
CUDA_VEC_FUNC(log, double)

#define DEFINE_ARRAY_FUNC_WRAPPER(type, func)                                                                      \
void array_##func##_##type(type *a, int size, int stride_in, type *out, int stride_out) {                          \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                                           \
    vector_##func##_##type<<<gridSize, BLOCK_SIZE>>>(a, size, stride_in, out, stride_out);                         \
}                                                                                                                  \
void array_##func##_assign_##type(type *a, int size, int stride) {                                                 \
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;                                                           \
    vector_##func##_assign_##type<<<gridSize, BLOCK_SIZE>>>(a, size, stride);                                      \
}

DEFINE_ARRAY_FUNC_WRAPPER(float, sin)
DEFINE_ARRAY_FUNC_WRAPPER(float, cos)
DEFINE_ARRAY_FUNC_WRAPPER(float, tan)
DEFINE_ARRAY_FUNC_WRAPPER(float, asin)
DEFINE_ARRAY_FUNC_WRAPPER(float, acos)
DEFINE_ARRAY_FUNC_WRAPPER(float, atan)
DEFINE_ARRAY_FUNC_WRAPPER(float, sinh)
DEFINE_ARRAY_FUNC_WRAPPER(float, cosh)
DEFINE_ARRAY_FUNC_WRAPPER(float, tanh)
DEFINE_ARRAY_FUNC_WRAPPER(float, abs)
DEFINE_ARRAY_FUNC_WRAPPER(float, sqrt)
DEFINE_ARRAY_FUNC_WRAPPER(float, exp)
DEFINE_ARRAY_FUNC_WRAPPER(float, log)

DEFINE_ARRAY_FUNC_WRAPPER(double, exp)
DEFINE_ARRAY_FUNC_WRAPPER(double, sin)
DEFINE_ARRAY_FUNC_WRAPPER(double, cos)
DEFINE_ARRAY_FUNC_WRAPPER(double, tan)
DEFINE_ARRAY_FUNC_WRAPPER(double, asin)
DEFINE_ARRAY_FUNC_WRAPPER(double, acos)
DEFINE_ARRAY_FUNC_WRAPPER(double, atan)
DEFINE_ARRAY_FUNC_WRAPPER(double, sinh)
DEFINE_ARRAY_FUNC_WRAPPER(double, cosh)
DEFINE_ARRAY_FUNC_WRAPPER(double, tanh)
DEFINE_ARRAY_FUNC_WRAPPER(double, abs)
DEFINE_ARRAY_FUNC_WRAPPER(double, sqrt) 
DEFINE_ARRAY_FUNC_WRAPPER(double, log)

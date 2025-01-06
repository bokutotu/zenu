/**
 * @file zenu_arith_nvidia.cu
 * @brief "nvidia" GPU implementations of the arithmetic functions.
 *
 * メモリ転送は行わず、入出力ポインタ(dst, src1, src2等)は
 * すべてGPU上のデバイスメモリを指していると仮定。
 * （ホスト側メモリへのコピーや割り当ては一切行わない。）
 *
 */

#include "zenu_compute.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdio.h>

//---------------------------------------------
// kernel launch config
//---------------------------------------------
static const int THREADS_PER_BLOCK = 256;

//---------------------------------------------
// GPU kernels
//   1) BinaryOpKernel : dst[i] = op(s1[i], s2[i])
//   2) ScalarOpKernel : dst[i] = op(s[i], c)
//   3) AssignOpKernel : dst[i] = op(dst[i], s[i])
//   4) AssignScalarOpKernel : dst[i] = op(dst[i], c)
//---------------------------------------------
template<typename T, typename OP>
__global__ void BinaryOpKernel(
    T* dst, const T* s1, const T* s2,
    int sd, int ss1, int ss2,
    size_t n, OP op)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        dst[idx * sd] = op(s1[idx * ss1], s2[idx * ss2]);
    }
}

template<typename T, typename OP>
__global__ void ScalarOpKernel(
    T* dst, const T* s,
    int sd, int ss,
    const T* c, size_t n, OP op)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        dst[idx * sd] = op(s[idx * ss], c[0]);
    }
}

template<typename T, typename OP>
__global__ void AssignOpKernel(
    T* dst, const T* s,
    int sd, int ss,
    size_t n, OP op)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        dst[idx * sd] = op(dst[idx * sd], s[idx * ss]);
    }
}

template<typename T, typename OP>
__global__ void AssignScalarOpKernel(
    T* dst,
    int sd,
    const T* c,
    size_t n,
    OP op)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        dst[idx * sd] = op(dst[idx * sd], c[0]);
    }
}

//---------------------------------------------
// Functors for +, -, *, /
//---------------------------------------------
struct AddOp {
    __device__ __host__ float  operator()(float a, float b)  const { return a + b; }
    __device__ __host__ double operator()(double a, double b)const { return a + b; }
};
struct SubOp {
    __device__ __host__ float  operator()(float a, float b)  const { return a - b; }
    __device__ __host__ double operator()(double a, double b)const { return a - b; }
};
struct MulOp {
    __device__ __host__ float  operator()(float a, float b)  const { return a * b; }
    __device__ __host__ double operator()(double a, double b)const { return a * b; }
};
struct DivOp {
    __device__ __host__ float  operator()(float a, float b)  const { return a / b; }
    __device__ __host__ double operator()(double a, double b)const { return a / b; }
};

//---------------------------------------------
// GPU呼び出し用ラッパーマクロ
//
// ここでは、ポインタ(dst, s1, s2等)は既にGPUメモリ上にあると仮定。
// strideは要素単位であると仮定。
//---------------------------------------------
#define ZENU_NVIDIA_BINARY_IMPL(FUNC_NAME, OP_FUNCTOR)                          \
ZenuStatus FUNC_NAME(                                                           \
    void* dst, const void* s1, const void* s2,                                  \
    int sd, int ss1, int ss2, size_t n, ZenuDataType dt)                        \
{                                                                               \
    if (!dst || !s1 || !s2) return InvalidArgument;                             \
    if (n == 0) return Success;                                                 \
    /* kernel launch */                                                         \
    dim3 block(THREADS_PER_BLOCK);                                              \
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));                     \
    if (dt == f32) {                                                            \
        BinaryOpKernel<float, OP_FUNCTOR><<<grid, block>>>(                     \
            (float*)dst, (const float*)s1, (const float*)s2,                    \
            sd, ss1, ss2, n, OP_FUNCTOR());                                     \
    } else {                                                                    \
        BinaryOpKernel<double, OP_FUNCTOR><<<grid, block>>>(                    \
            (double*)dst, (const double*)s1, (const double*)s2,                 \
            sd, ss1, ss2, n, OP_FUNCTOR());                                     \
    }                                                                           \
    cudaError_t e = cudaDeviceSynchronize();                                    \
    return convertCudaError(e);                                                \
}

#define ZENU_NVIDIA_SCALAR_IMPL(FUNC_NAME, OP_FUNCTOR)                          \
ZenuStatus FUNC_NAME(                                                           \
    void* dst, const void* s, int sd, int ss,                                   \
    const void* sc, size_t n, ZenuDataType dt)                                  \
{                                                                               \
    if (!dst || !s || !sc) return InvalidArgument;                              \
    if (n == 0) return Success;                                                 \
    dim3 block(THREADS_PER_BLOCK);                                              \
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));                     \
    if (dt == f32) {                                                            \
        ScalarOpKernel<float, OP_FUNCTOR><<<grid, block>>>(                     \
            (float*)dst, (const float*)s,                                       \
            sd, ss, (const float*)sc, n, OP_FUNCTOR());                         \
    } else {                                                                    \
        ScalarOpKernel<double, OP_FUNCTOR><<<grid, block>>>(                    \
            (double*)dst, (const double*)s,                                     \
            sd, ss, (const double*)sc, n, OP_FUNCTOR());                        \
    }                                                                           \
    cudaError_t e = cudaDeviceSynchronize();                                    \
    return convertCudaError(e);                                                \
}

#define ZENU_NVIDIA_ASSIGN_IMPL(FUNC_NAME, OP_FUNCTOR)                          \
ZenuStatus FUNC_NAME(                                                           \
    void* dst, const void* s, int sd, int ss, size_t n, ZenuDataType dt)        \
{                                                                               \
    if (!dst || !s) return InvalidArgument;                                     \
    if (n == 0) return Success;                                                 \
    dim3 block(THREADS_PER_BLOCK);                                              \
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));                     \
    if (dt == f32) {                                                            \
        AssignOpKernel<float, OP_FUNCTOR><<<grid, block>>>(                     \
            (float*)dst, (const float*)s, sd, ss, n, OP_FUNCTOR());            \
    } else {                                                                    \
        AssignOpKernel<double, OP_FUNCTOR><<<grid, block>>>(                    \
            (double*)dst, (const double*)s, sd, ss, n, OP_FUNCTOR());          \
    }                                                                           \
    cudaError_t e = cudaDeviceSynchronize();                                    \
    return convertCudaError(e);                                                \
}

#define ZENU_NVIDIA_ASSIGN_SCALAR_IMPL(FUNC_NAME, OP_FUNCTOR)                   \
ZenuStatus FUNC_NAME(                                                           \
    void* dst, int sd, const void* sc, size_t n, ZenuDataType dt)               \
{                                                                               \
    if (!dst || !sc) return InvalidArgument;                                    \
    if (n == 0) return Success;                                                 \
    dim3 block(THREADS_PER_BLOCK);                                              \
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));                     \
    if (dt == f32) {                                                            \
        AssignScalarOpKernel<float, OP_FUNCTOR><<<grid, block>>>(               \
            (float*)dst, sd, (const float*)sc, n, OP_FUNCTOR());                \
    } else {                                                                    \
        AssignScalarOpKernel<double, OP_FUNCTOR><<<grid, block>>>(              \
            (double*)dst, sd, (const double*)sc, n, OP_FUNCTOR());              \
    }                                                                           \
    cudaError_t e = cudaDeviceSynchronize();                                    \
    return convertCudaError(e);                                                \
}

//---------------------------------------------
// ADD
//---------------------------------------------
ZENU_NVIDIA_BINARY_IMPL(zenu_compute_add_mat_mat_nvidia,            AddOp)
ZENU_NVIDIA_SCALAR_IMPL(zenu_compute_add_mat_scalar_ptr_nvidia,     AddOp)
ZENU_NVIDIA_ASSIGN_IMPL(zenu_compute_add_mat_mat_assign_nvidia,     AddOp)
ZENU_NVIDIA_ASSIGN_SCALAR_IMPL(zenu_compute_add_mat_scalar_ptr_assign_nvidia, AddOp)

//---------------------------------------------
// SUB
//---------------------------------------------
ZENU_NVIDIA_BINARY_IMPL(zenu_compute_sub_mat_mat_nvidia,            SubOp)
ZENU_NVIDIA_SCALAR_IMPL(zenu_compute_sub_mat_scalar_ptr_nvidia,     SubOp)
ZENU_NVIDIA_ASSIGN_IMPL(zenu_compute_sub_mat_mat_assign_nvidia,     SubOp)
ZENU_NVIDIA_ASSIGN_SCALAR_IMPL(zenu_compute_sub_mat_scalar_ptr_assign_nvidia, SubOp)

//---------------------------------------------
// MUL
//---------------------------------------------
ZENU_NVIDIA_BINARY_IMPL(zenu_compute_mul_mat_mat_nvidia,            MulOp)
ZENU_NVIDIA_SCALAR_IMPL(zenu_compute_mul_mat_scalar_ptr_nvidia,     MulOp)
ZENU_NVIDIA_ASSIGN_IMPL(zenu_compute_mul_mat_mat_assign_nvidia,     MulOp)
ZENU_NVIDIA_ASSIGN_SCALAR_IMPL(zenu_compute_mul_mat_scalar_ptr_assign_nvidia, MulOp)

//---------------------------------------------
// DIV
//---------------------------------------------
ZENU_NVIDIA_BINARY_IMPL(zenu_compute_div_mat_mat_nvidia,            DivOp)
ZENU_NVIDIA_SCALAR_IMPL(zenu_compute_div_mat_scalar_ptr_nvidia,     DivOp)
ZENU_NVIDIA_ASSIGN_IMPL(zenu_compute_div_mat_mat_assign_nvidia,     DivOp)
ZENU_NVIDIA_ASSIGN_SCALAR_IMPL(zenu_compute_div_mat_scalar_ptr_assign_nvidia, DivOp)

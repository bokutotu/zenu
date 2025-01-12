#pragma once

#include "zenu_compute.h"
#include "utils.h"

#include <cuda_runtime.h>

//=====================================================
//  NVIDIA 実装 (CUDA) 用のマクロ群
//=====================================================
static const int THREADS_PER_BLOCK = 256;

/*-----------------------------------------
 * 2 入力演算 (binary op)
 *   dst[i] = src1[i] <op> src2[i]
 *-----------------------------------------*/
#define ZENU_NVIDIA_BINARY_OP(FUNC_NAME, OP_FUNCTOR)                                 \
ZenuStatus FUNC_NAME(                                                             \
    void* dst, const void* s1, const void* s2,                                    \
    int sd, int ss1, int ss2, size_t n, ZenuDataType dt)                          \
{                                                                                 \
    if (!dst || !s1 || !s2) return InvalidArgument;                               \
    if (n == 0) return Success;                                                   \
    dim3 block(THREADS_PER_BLOCK);                                                \
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));                       \
    if (dt == f32) {                                                              \
        BinaryOpKernel<float, OP_FUNCTOR><<<grid, block>>>(                      \
            (float*)dst, (const float*)s1, (const float*)s2,                      \
            sd, ss1, ss2, n, OP_FUNCTOR());                                       \
    } else if (dt == f64) {                                                       \
        BinaryOpKernel<double, OP_FUNCTOR><<<grid, block>>>(                     \
            (double*)dst, (const double*)s1, (const double*)s2,                   \
            sd, ss1, ss2, n, OP_FUNCTOR());                                       \
    } else {                                                                      \
        return InvalidArgument;                                                   \
    }                                                                             \
    cudaError_t e = cudaDeviceSynchronize();                                      \
    return convertCudaError(e);                                                  \
}

/*-----------------------------------------
 * スカラー演算 (scalar op)
 *   dst[i] = src[i] <op> c
 *-----------------------------------------*/
#define ZENU_NVIDIA_SCALAR_OP(FUNC_NAME, OP_FUNCTOR)                                 \
ZenuStatus FUNC_NAME(                                                             \
    void* dst, const void* s, int sd, int ss,                                     \
    const void* sc, size_t n, ZenuDataType dt)                                    \
{                                                                                 \
    if (!dst || !s || !sc) return InvalidArgument;                                \
    if (n == 0) return Success;                                                   \
    dim3 block(THREADS_PER_BLOCK);                                                \
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));                       \
    if (dt == f32) {                                                              \
        ScalarOpKernel<float, OP_FUNCTOR><<<grid, block>>>(                      \
            (float*)dst, (const float*)s, sd, ss, (const float*)sc, n, OP_FUNCTOR()); \
    } else if (dt == f64) {                                                       \
        ScalarOpKernel<double, OP_FUNCTOR><<<grid, block>>>(                     \
            (double*)dst, (const double*)s, sd, ss, (const double*)sc, n, OP_FUNCTOR());\
    } else {                                                                      \
        return InvalidArgument;                                                   \
    }                                                                             \
    cudaError_t e = cudaDeviceSynchronize();                                      \
    return convertCudaError(e);                                                  \
}

/*-----------------------------------------
 * 複合代入 (assign op)
 *   dst[i] <op>= src[i]
 *-----------------------------------------*/
#define ZENU_NVIDIA_ASSIGN_OP(FUNC_NAME, OP_FUNCTOR)                                 \
ZenuStatus FUNC_NAME(                                                             \
    void* dst, const void* s, int sd, int ss, size_t n, ZenuDataType dt)          \
{                                                                                 \
    if (!dst || !s) return InvalidArgument;                                       \
    if (n == 0) return Success;                                                   \
    dim3 block(THREADS_PER_BLOCK);                                                \
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));                       \
    if (dt == f32) {                                                              \
        AssignOpKernel<float, OP_FUNCTOR><<<grid, block>>>(                      \
            (float*)dst, (const float*)s, sd, ss, n, OP_FUNCTOR());              \
    } else if (dt == f64) {                                                       \
        AssignOpKernel<double, OP_FUNCTOR><<<grid, block>>>(                     \
            (double*)dst, (const double*)s, sd, ss, n, OP_FUNCTOR());            \
    } else {                                                                      \
        return InvalidArgument;                                                   \
    }                                                                             \
    cudaError_t e = cudaDeviceSynchronize();                                      \
    return convertCudaError(e);                                                  \
}

/*-----------------------------------------
 * スカラー複合代入 (assign scalar op)
 *   dst[i] <op>= c
 *-----------------------------------------*/
#define ZENU_NVIDIA_ASSIGN_SCALAR_OP(FUNC_NAME, OP_FUNCTOR)                          \
ZenuStatus FUNC_NAME(                                                             \
    void* dst, int sd, const void* sc, size_t n, ZenuDataType dt)                 \
{                                                                                 \
    if (!dst || !sc) return InvalidArgument;                                      \
    if (n == 0) return Success;                                                   \
    dim3 block(THREADS_PER_BLOCK);                                                \
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));                       \
    if (dt == f32) {                                                              \
        AssignScalarOpKernel<float, OP_FUNCTOR><<<grid, block>>>(                \
            (float*)dst, sd, (const float*)sc, n, OP_FUNCTOR());                 \
    } else if (dt == f64) {                                                       \
        AssignScalarOpKernel<double, OP_FUNCTOR><<<grid, block>>>(               \
            (double*)dst, sd, (const double*)sc, n, OP_FUNCTOR());               \
    } else {                                                                      \
        return InvalidArgument;                                                   \
    }                                                                             \
    cudaError_t e = cudaDeviceSynchronize();                                      \
    return convertCudaError(e);                                                  \
}

template<typename T, typename OP>
__global__ void UnaryOpKernel(T* dst, const T* src, int sd, int ss, size_t n, OP op)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[sd * i] = op(src[ss * i]);
    }
}

template<typename T, typename OP>
__global__ void UnaryAssignOpKernel(T* dst, int sd, size_t n, OP op)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[sd * i] = op(dst[sd * i]);
    }
}

/*-----------------------------------------
 * 単項演算 (unary op)
 *   dst[i] = f( src[i] )
 *-----------------------------------------*/
#define ZENU_NVIDIA_UNARY_OP(FUNC_NAME, OP_FUNCTOR)                                  \
ZenuStatus FUNC_NAME(                                                             \
    void* dst, const void* s, int sd, int ss, size_t n, ZenuDataType dt)          \
{                                                                                 \
    if (!dst || !s) return InvalidArgument;                                       \
    if (n == 0) return Success;                                                   \
    dim3 block(THREADS_PER_BLOCK);                                                \
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));                       \
    if (dt == f32) {                                                              \
        UnaryOpKernel<float, OP_FUNCTOR><<<grid, block>>>(                       \
            (float*)dst, (const float*)s, sd, ss, n, OP_FUNCTOR());              \
    } else if (dt == f64) {                                                       \
        UnaryOpKernel<double, OP_FUNCTOR><<<grid, block>>>(                      \
            (double*)dst, (const double*)s, sd, ss, n, OP_FUNCTOR());            \
    } else {                                                                      \
        return InvalidArgument;                                                   \
    }                                                                             \
    cudaError_t e = cudaDeviceSynchronize();                                      \
    return convertCudaError(e);                                                  \
}

/*-----------------------------------------
 * 単項演算 (unary op, in-place)
 *   dst[i] = f( dst[i] )
 *-----------------------------------------*/
#define ZENU_NVIDIA_UNARY_ASSIGN_OP(FUNC_NAME, OP_FUNCTOR)                           \
ZenuStatus FUNC_NAME(                                                             \
    void* dst, int sd, size_t n, ZenuDataType dt)                                 \
{                                                                                 \
    if (!dst) return InvalidArgument;                                             \
    if (n == 0) return Success;                                                   \
    dim3 block(THREADS_PER_BLOCK);                                                \
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));                       \
    if (dt == f32) {                                                              \
        UnaryAssignOpKernel<float, OP_FUNCTOR><<<grid, block>>>(                 \
            (float*)dst, sd, n, OP_FUNCTOR());                                   \
    } else if (dt == f64) {                                                       \
        UnaryAssignOpKernel<double, OP_FUNCTOR><<<grid, block>>>(                \
            (double*)dst, sd, n, OP_FUNCTOR());                                  \
    } else {                                                                      \
        return InvalidArgument;                                                   \
    }                                                                             \
    cudaError_t e = cudaDeviceSynchronize();                                      \
    return convertCudaError(e);                                                  \
}


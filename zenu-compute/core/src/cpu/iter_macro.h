#pragma once
#include "zenu_compute.h"
#include "utils.h"

#include <omp.h>
#include <string.h>
#include <stdio.h>

/*-----------------------------------------
 * 1) 2入力演算 (binary op)
 *    dst[i] = OP_FUNCTOR()( src1[i], src2[i] )
 *-----------------------------------------*/
#define ZENU_CPU_BINARY_OP(FUNC_NAME, OP_FUNCTOR)                                     \
ZenuStatus FUNC_NAME(                                                                 \
    void* dst, const void* src1, const void* src2,                                    \
    int stride_dst, int stride_src1, int stride_src2,                                 \
    size_t n, ZenuDataType dt)                                                        \
{                                                                                     \
    if (!dst || !src1 || !src2) return InvalidArgument;                               \
    if (n == 0) return Success;                                                       \
    if (dt == f32) {                                                                  \
        float*       pDst  = (float*)dst;                                            \
        const float* pSrc1 = (const float*)src1;                                      \
        const float* pSrc2 = (const float*)src2;                                      \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()(                                     \
                pSrc1[i * stride_src1],                                              \
                pSrc2[i * stride_src2]);                                             \
        }                                                                             \
    } else if (dt == f64) {                                                           \
        double*       pDst  = (double*)dst;                                          \
        const double* pSrc1 = (const double*)src1;                                    \
        const double* pSrc2 = (const double*)src2;                                    \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()(                                     \
                pSrc1[i * stride_src1],                                              \
                pSrc2[i * stride_src2]);                                             \
        }                                                                             \
    } else {                                                                          \
        return InvalidArgument;                                                       \
    }                                                                                 \
    return Success;                                                                   \
}

/*-----------------------------------------
 * 2) スカラー演算 (scalar op)
 *    dst[i] = OP_FUNCTOR()( src[i], c )
 *-----------------------------------------*/
#define ZENU_CPU_SCALAR_OP(FUNC_NAME, OP_FUNCTOR)                                     \
ZenuStatus FUNC_NAME(                                                                 \
    void* dst, const void* src,                                                      \
    int stride_dst, int stride_src,                                                  \
    const void* scalar_ptr,                                                          \
    size_t n, ZenuDataType dt)                                                       \
{                                                                                     \
    if (!dst || !src || !scalar_ptr) return InvalidArgument;                         \
    if (n == 0) return Success;                                                       \
    if (dt == f32) {                                                                  \
        float*       pDst = (float*)dst;                                             \
        const float* pSrc = (const float*)src;                                       \
        float        c    = *(const float*)scalar_ptr;                               \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()(                                     \
                pSrc[i * stride_src], c );                                           \
        }                                                                             \
    } else if (dt == f64) {                                                           \
        double*       pDst = (double*)dst;                                           \
        const double* pSrc = (const double*)src;                                     \
        double        c    = *(const double*)scalar_ptr;                             \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()(                                     \
                pSrc[i * stride_src], c );                                           \
        }                                                                             \
    } else {                                                                          \
        return InvalidArgument;                                                       \
    }                                                                                 \
    return Success;                                                                   \
}

/*-----------------------------------------
 * 3) 複合代入 (assign op)
 *    dst[i] = OP_FUNCTOR()( dst[i], src[i] )
 *-----------------------------------------*/
#define ZENU_CPU_ASSIGN_OP(FUNC_NAME, OP_FUNCTOR)                                     \
ZenuStatus FUNC_NAME(                                                                 \
    void* dst, const void* src,                                                      \
    int stride_dst, int stride_src,                                                  \
    size_t n, ZenuDataType dt)                                                       \
{                                                                                     \
    if (!dst || !src) return InvalidArgument;                                        \
    if (n == 0) return Success;                                                       \
    if (dt == f32) {                                                                  \
        float*       pDst = (float*)dst;                                             \
        const float* pSrc = (const float*)src;                                       \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()(                                     \
                pDst[i * stride_dst], pSrc[i * stride_src]);                         \
        }                                                                             \
    } else if (dt == f64) {                                                           \
        double*       pDst = (double*)dst;                                           \
        const double* pSrc = (const double*)src;                                     \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()(                                     \
                pDst[i * stride_dst], pSrc[i * stride_src]);                         \
        }                                                                             \
    } else {                                                                          \
        return InvalidArgument;                                                       \
    }                                                                                 \
    return Success;                                                                   \
}

/*-----------------------------------------
 * 4) スカラー複合代入 (assign scalar op)
 *    dst[i] = OP_FUNCTOR()( dst[i], c )
 *-----------------------------------------*/
#define ZENU_CPU_ASSIGN_SCALAR_OP(FUNC_NAME, OP_FUNCTOR)                              \
ZenuStatus FUNC_NAME(                                                                 \
    void* dst, int stride_dst,                                                       \
    const void* scalar_ptr,                                                          \
    size_t n, ZenuDataType dt)                                                       \
{                                                                                     \
    if (!dst || !scalar_ptr) return InvalidArgument;                                 \
    if (n == 0) return Success;                                                       \
    if (dt == f32) {                                                                  \
        float* pDst = (float*)dst;                                                   \
        float  c    = *(const float*)scalar_ptr;                                     \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()(                                     \
                pDst[i * stride_dst], c );                                           \
        }                                                                             \
    } else if (dt == f64) {                                                           \
        double* pDst = (double*)dst;                                                 \
        double  c    = *(const double*)scalar_ptr;                                   \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()(                                     \
                pDst[i * stride_dst], c );                                           \
        }                                                                             \
    } else {                                                                          \
        return InvalidArgument;                                                       \
    }                                                                                 \
    return Success;                                                                   \
}

/*-----------------------------------------
 * 5) 単項演算 (unary op)
 *    dst[i] = OP_FUNCTOR()( src[i] )
 *-----------------------------------------*/
#define ZENU_CPU_UNARY_OP(FUNC_NAME, OP_FUNCTOR)                                      \
ZenuStatus FUNC_NAME(                                                                 \
    void* dst, const void* src,                                                      \
    int stride_dst, int stride_src,                                                  \
    size_t n, ZenuDataType dt)                                                       \
{                                                                                     \
    if (!dst || !src) return InvalidArgument;                                        \
    if (n == 0) return Success;                                                       \
    if (dt == f32) {                                                                  \
        float*       pDst = (float*)dst;                                             \
        const float* pSrc = (const float*)src;                                       \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()( pSrc[i * stride_src] );             \
        }                                                                             \
    } else if (dt == f64) {                                                           \
        double*       pDst = (double*)dst;                                           \
        const double* pSrc = (const double*)src;                                     \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()( pSrc[i * stride_src] );             \
        }                                                                             \
    } else {                                                                          \
        return InvalidArgument;                                                       \
    }                                                                                 \
    return Success;                                                                   \
}

/*-----------------------------------------
 * 6) 単項演算 (unary op, in-place)
 *    dst[i] = OP_FUNCTOR()( dst[i] )
 *-----------------------------------------*/
#define ZENU_CPU_UNARY_ASSIGN_OP(FUNC_NAME, OP_FUNCTOR)                               \
ZenuStatus FUNC_NAME(                                                                 \
    void* dst, int stride_dst,                                                       \
    size_t n, ZenuDataType dt)                                                       \
{                                                                                     \
    if (!dst) return InvalidArgument;                                                \
    if (n == 0) return Success;                                                       \
    if (dt == f32) {                                                                  \
        float* pDst = (float*)dst;                                                   \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()( pDst[i * stride_dst] );             \
        }                                                                             \
    } else if (dt == f64) {                                                           \
        double* pDst = (double*)dst;                                                 \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = OP_FUNCTOR()( pDst[i * stride_dst] );             \
        }                                                                             \
    } else {                                                                          \
        return InvalidArgument;                                                       \
    }                                                                                 \
    return Success;                                                                   \
}


#pragma once
#include "zenu_compute.h"
#include "utils.h"

#include <omp.h>
#include <string.h>
#include <stdio.h>

/*-----------------------------------------
 * 2 入力演算 (binary op)
 *   dst[i] = src1[i] <op> src2[i]
 *-----------------------------------------*/
#define ZENU_CPU_BINARY_OP(FUNC_NAME, OP, TYPE)                                        \
ZenuStatus FUNC_NAME(                                                                  \
    void* dst, const void* src1, const void* src2,                                     \
    int stride_dst, int stride_src1, int stride_src2,                                  \
    size_t n, ZenuDataType data_type)                                                  \
{                                                                                      \
    ZenuStatus st = check_common_args(dst, n, data_type);                             \
    if (st != Success) return st;                                                      \
    if (!src1 || !src2) return InvalidArgument;                                        \
    if (n == 0) return Success;                                                        \
    TYPE*       pDst  = (TYPE*)dst;                                                   \
    const TYPE* pSrc1 = (const TYPE*)src1;                                             \
    const TYPE* pSrc2 = (const TYPE*)src2;                                             \
    _Pragma("omp parallel for simd")                                                  \
    for (size_t i = 0; i < n; i++) {                                                  \
        pDst[i * stride_dst] = pSrc1[i * stride_src1] OP pSrc2[i * stride_src2];       \
    }                                                                                  \
    return Success;                                                                    \
}

/*-----------------------------------------
 * スカラー演算 (scalar op)
 *   dst[i] = src[i] <op> c
 *-----------------------------------------*/
#define ZENU_CPU_SCALAR_OP(FUNC_NAME, OP, TYPE)                                       \
ZenuStatus FUNC_NAME(                                                                 \
    void* dst, const void* src,                                                       \
    int stride_dst, int stride_src,                                                   \
    const void* scalar_ptr,                                                           \
    size_t n, ZenuDataType data_type)                                                 \
{                                                                                     \
    ZenuStatus st = check_common_args(dst, n, data_type);                            \
    if (st != Success) return st;                                                     \
    if (!src || !scalar_ptr) return InvalidArgument;                                  \
    if (n == 0) return Success;                                                       \
    TYPE*       pDst = (TYPE*)dst;                                                   \
    const TYPE* pSrc = (const TYPE*)src;                                             \
    const TYPE  c    = *(const TYPE*)scalar_ptr;                                      \
    _Pragma("omp parallel for simd")                                                 \
    for (size_t i = 0; i < n; i++) {                                                 \
        pDst[i * stride_dst] = pSrc[i * stride_src] OP c;                            \
    }                                                                                 \
    return Success;                                                                   \
}

/*-----------------------------------------
 * 複合代入 (assign op)
 *   dst[i] <op>= src[i]
 *-----------------------------------------*/
#define ZENU_CPU_ASSIGN_OP(FUNC_NAME, OP, TYPE)                                       \
ZenuStatus FUNC_NAME(                                                                 \
    void* dst, const void* src,                                                       \
    int stride_dst, int stride_src,                                                   \
    size_t n, ZenuDataType data_type)                                                 \
{                                                                                     \
    ZenuStatus st = check_common_args(dst, n, data_type);                            \
    if (st != Success) return st;                                                     \
    if (!src) return InvalidArgument;                                                 \
    if (n == 0) return Success;                                                       \
    TYPE*       pDst = (TYPE*)dst;                                                   \
    const TYPE* pSrc = (const TYPE*)src;                                             \
    _Pragma("omp parallel for simd")                                                 \
    for (size_t i = 0; i < n; i++) {                                                 \
        pDst[i * stride_dst] OP pSrc[i * stride_src];                                \
    }                                                                                 \
    return Success;                                                                   \
}

/*-----------------------------------------
 * スカラー複合代入 (assign scalar op)
 *   dst[i] <op>= c
 *-----------------------------------------*/
#define ZENU_CPU_ASSIGN_SCALAR_OP(FUNC_NAME, OP, TYPE)                                \
ZenuStatus FUNC_NAME(                                                                 \
    void* dst, int stride_dst,                                                        \
    const void* scalar_ptr,                                                           \
    size_t n, ZenuDataType data_type)                                                 \
{                                                                                     \
    ZenuStatus st = check_common_args(dst, n, data_type);                            \
    if (st != Success) return st;                                                     \
    if (!scalar_ptr) return InvalidArgument;                                          \
    if (n == 0) return Success;                                                       \
    TYPE* pDst = (TYPE*)dst;                                                         \
    const TYPE c = *(const TYPE*)scalar_ptr;                                          \
    _Pragma("omp parallel for simd")                                                 \
    for (size_t i = 0; i < n; i++) {                                                 \
        pDst[i * stride_dst] OP c;                                                   \
    }                                                                                 \
    return Success;                                                                   \
}

/*-----------------------------------------
 * 単項演算 (unary op)
 *   dst[i] = f( src[i] )
 *-----------------------------------------*/
#define ZENU_CPU_UNARY_OP(FUNC_NAME, MATH_FUNC_F32, MATH_FUNC_F64)                    \
ZenuStatus FUNC_NAME(                                                                 \
    void* dst, const void* src,                                                       \
    int stride_dst, int stride_src,                                                   \
    size_t n, ZenuDataType data_type)                                                 \
{                                                                                     \
    ZenuStatus st = check_common_args(dst, n, data_type);                            \
    if (st != Success) return st;                                                     \
    if (!src) return InvalidArgument;                                                 \
    if (n == 0) return Success;                                                       \
    if (data_type == f32) {                                                           \
        float*       pDst = (float*)dst;                                             \
        const float* pSrc = (const float*)src;                                       \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = MATH_FUNC_F32(pSrc[i * stride_src]);              \
        }                                                                             \
    } else { /* f64 */                                                               \
        double*       pDst = (double*)dst;                                           \
        const double* pSrc = (const double*)src;                                     \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = MATH_FUNC_F64(pSrc[i * stride_src]);              \
        }                                                                             \
    }                                                                                 \
    return Success;                                                                   \
}

/*-----------------------------------------
 * 単項演算 (unary op, in-place)
 *   dst[i] = f( dst[i] )
 *-----------------------------------------*/
#define ZENU_CPU_UNARY_ASSIGN_OP(FUNC_NAME, MATH_FUNC_F32, MATH_FUNC_F64)             \
ZenuStatus FUNC_NAME(                                                                 \
    void* dst, int stride_dst,                                                        \
    size_t n, ZenuDataType data_type)                                                 \
{                                                                                     \
    ZenuStatus st = check_common_args(dst, n, data_type);                            \
    if (st != Success) return st;                                                     \
    if (n == 0) return Success;                                                       \
    if (data_type == f32) {                                                           \
        float* pDst = (float*)dst;                                                   \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = MATH_FUNC_F32(pDst[i * stride_dst]);              \
        }                                                                             \
    } else { /* f64 */                                                               \
        double* pDst = (double*)dst;                                                 \
        _Pragma("omp parallel for simd")                                             \
        for (size_t i = 0; i < n; i++) {                                             \
            pDst[i * stride_dst] = MATH_FUNC_F64(pDst[i * stride_dst]);              \
        }                                                                             \
    }                                                                                 \
    return Success;                                                                   \
}


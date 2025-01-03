/**
 * @file arithmetic.cpp
 * @brief Implementation of arithmetic functions (add/sub/mul/div) on CPU / "nvidia" GPU.
 *
 *        This is "製品コード" with emphasis on performance:
 *        - Uses OpenMP for multi-threading.
 *        - Designed for auto-vectorization (-march, -O3, etc.).
 *        - Uses macros to keep code concise while covering many variants.
 *
 *        The "nvidia" versions here are placeholders and return `DeviceError`
 *        (or you could implement them with an actual GPU backend).
 */

#include "zenu_compute.h"
#include <omp.h>
#include <string.h> // (optional) for memset etc. if needed
#include <stdio.h>  // (optional) for debug prints if needed

/*=========================================================
 * ヘルパー: 引数チェック
 *========================================================*/
static inline ZenuStatus
check_common_args(const void* dst, size_t n, ZenuDataType dt)
{
    if (!dst)                 return InvalidArgument;
    if (dt != f32 && dt != f64) return InvalidArgument;
    /* n=0 は何もしなくてもよいので Success を返す */
    return (n == 0) ? Success : Success;
}

/*=========================================================
 * OpenMP + auto-vector化用マクロ
 *
 * - 2入力( dst[i] = src1[i] <op> src2[i] )
 * - スカラー演算( dst[i] = src[i] <op> c )
 * - 複合代入( dst[i] <op>= src[i] )
 * - スカラー複合代入( dst[i] <op>= c )
 *
 * stride_* は要素単位（float or double）でカウントされているものとする。
 *========================================================*/

/*---------------------------
 * 2入力演算
 *---------------------------*/
#define ZENU_CPU_BINARY_OP(FUNC_NAME, OP, TYPE)                               \
    ZenuStatus FUNC_NAME(                                                     \
        void*       dst,                                                     \
        const void* src1,                                                    \
        const void* src2,                                                    \
        int         stride_dst,                                              \
        int         stride_src1,                                             \
        int         stride_src2,                                             \
        size_t      n,                                                       \
        ZenuDataType data_type)                                              \
    {                                                                         \
        ZenuStatus st = check_common_args(dst, n, data_type);                \
        if (st != Success) return st;                                         \
        if (!src1 || !src2) return InvalidArgument;                           \
        if (n == 0) return Success;                                           \
        TYPE*       pDst  = (TYPE*)dst;                                      \
        const TYPE* pSrc1 = (const TYPE*)src1;                                \
        const TYPE* pSrc2 = (const TYPE*)src2;                                \
        /* OpenMP + auto-vectorization */                                     \
        _Pragma("omp parallel for simd")                                      \
        for (size_t i = 0; i < n; i++) {                                      \
            pDst[i * stride_dst] = pSrc1[i * stride_src1] OP pSrc2[i * stride_src2]; \
        }                                                                     \
        return Success;                                                       \
    }

/*---------------------------
 * スカラー演算
 *---------------------------*/
#define ZENU_CPU_SCALAR_OP(FUNC_NAME, OP, TYPE)                              \
    ZenuStatus FUNC_NAME(                                                    \
        void*       dst,                                                     \
        const void* src,                                                     \
        int         stride_dst,                                              \
        int         stride_src,                                              \
        const void* scalar_ptr,                                              \
        size_t      n,                                                       \
        ZenuDataType data_type)                                              \
    {                                                                        \
        ZenuStatus st = check_common_args(dst, n, data_type);               \
        if (st != Success) return st;                                        \
        if (!src || !scalar_ptr) return InvalidArgument;                     \
        if (n == 0) return Success;                                          \
        TYPE*       pDst = (TYPE*)dst;                                       \
        const TYPE* pSrc = (const TYPE*)src;                                 \
        const TYPE  c    = *(const TYPE*)scalar_ptr;                         \
        _Pragma("omp parallel for simd")                                     \
        for (size_t i = 0; i < n; i++) {                                     \
            pDst[i * stride_dst] = pSrc[i * stride_src] OP c;               \
        }                                                                    \
        return Success;                                                      \
    }

/*---------------------------
 * 複合代入
 *   dst[i] <op>= src[i]
 *---------------------------*/
#define ZENU_CPU_ASSIGN_OP(FUNC_NAME, OP, TYPE)                              \
    ZenuStatus FUNC_NAME(                                                    \
        void*       dst,                                                     \
        const void* src,                                                     \
        int         stride_dst,                                              \
        int         stride_src,                                              \
        size_t      n,                                                       \
        ZenuDataType data_type)                                              \
    {                                                                        \
        ZenuStatus st = check_common_args(dst, n, data_type);               \
        if (st != Success) return st;                                        \
        if (!src) return InvalidArgument;                                    \
        if (n == 0) return Success;                                          \
        TYPE*       pDst = (TYPE*)dst;                                       \
        const TYPE* pSrc = (const TYPE*)src;                                 \
        _Pragma("omp parallel for simd")                                     \
        for (size_t i = 0; i < n; i++) {                                     \
            pDst[i * stride_dst] OP pSrc[i * stride_src];                    \
        }                                                                    \
        return Success;                                                      \
    }

/*---------------------------
 * スカラー複合代入
 *   dst[i] <op>= c
 *   (src と stride_src は使わない)
 *---------------------------*/
#define ZENU_CPU_ASSIGN_SCALAR_OP(FUNC_NAME, OP, TYPE)                       \
    ZenuStatus FUNC_NAME(                                                    \
        void*       dst,                                                     \
        int         stride_dst,                                              \
        const void* scalar_ptr,                                              \
        size_t      n,                                                       \
        ZenuDataType data_type)                                              \
    {                                                                        \
        ZenuStatus st = check_common_args(dst, n, data_type);               \
        if (st != Success) return st;                                        \
        if (!scalar_ptr) return InvalidArgument;                             \
        if (n == 0) return Success;                                          \
        TYPE* pDst = (TYPE*)dst;                                            \
        const TYPE c = *(const TYPE*)scalar_ptr;                             \
        _Pragma("omp parallel for simd")                                     \
        for (size_t i = 0; i < n; i++) {                                     \
            pDst[i * stride_dst] OP c;                                       \
        }                                                                    \
        return Success;                                                      \
    }

/*=========================================================
 * ADD (CPU)
 *========================================================*/
ZENU_CPU_BINARY_OP(zenu_compute_add_mat_mat_cpu_f32, +, float)
ZENU_CPU_BINARY_OP(zenu_compute_add_mat_mat_cpu_f64, +, double)
/* ディスパッチ: data_typeに応じて */
ZenuStatus zenu_compute_add_mat_mat_cpu(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_add_mat_mat_cpu_f32(
            dst, src1, src2, stride_dst, stride_src1, stride_src2, n, f32);
    } else {
        return zenu_compute_add_mat_mat_cpu_f64(
            dst, src1, src2, stride_dst, stride_src1, stride_src2, n, f64);
    }
}

ZENU_CPU_SCALAR_OP(zenu_compute_add_mat_scalar_ptr_cpu_f32, +, float)
ZENU_CPU_SCALAR_OP(zenu_compute_add_mat_scalar_ptr_cpu_f64, +, double)
ZenuStatus zenu_compute_add_mat_scalar_ptr_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_add_mat_scalar_ptr_cpu_f32(
            dst, src, stride_dst, stride_src, scalar_ptr, n, f32);
    } else {
        return zenu_compute_add_mat_scalar_ptr_cpu_f64(
            dst, src, stride_dst, stride_src, scalar_ptr, n, f64);
    }
}

ZENU_CPU_ASSIGN_OP(zenu_compute_add_mat_mat_assign_cpu_f32, +=, float)
ZENU_CPU_ASSIGN_OP(zenu_compute_add_mat_mat_assign_cpu_f64, +=, double)
ZenuStatus zenu_compute_add_mat_mat_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_add_mat_mat_assign_cpu_f32(
            dst, src, stride_dst, stride_src, n, f32);
    } else {
        return zenu_compute_add_mat_mat_assign_cpu_f64(
            dst, src, stride_dst, stride_src, n, f64);
    }
}

ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_add_mat_scalar_ptr_assign_cpu_f32, +=, float)
ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_add_mat_scalar_ptr_assign_cpu_f64, +=, double)
ZenuStatus zenu_compute_add_mat_scalar_ptr_assign_cpu(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_add_mat_scalar_ptr_assign_cpu_f32(
            dst, stride_dst, scalar_ptr, n, f32);
    } else {
        return zenu_compute_add_mat_scalar_ptr_assign_cpu_f64(
            dst, stride_dst, scalar_ptr, n, f64);
    }
}

/*=========================================================
 * SUB (CPU)
 *========================================================*/
ZENU_CPU_BINARY_OP(zenu_compute_sub_mat_mat_cpu_f32, -, float)
ZENU_CPU_BINARY_OP(zenu_compute_sub_mat_mat_cpu_f64, -, double)
ZenuStatus zenu_compute_sub_mat_mat_cpu(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_sub_mat_mat_cpu_f32(
            dst, src1, src2, stride_dst, stride_src1, stride_src2, n, f32);
    } else {
        return zenu_compute_sub_mat_mat_cpu_f64(
            dst, src1, src2, stride_dst, stride_src1, stride_src2, n, f64);
    }
}

ZENU_CPU_SCALAR_OP(zenu_compute_sub_mat_scalar_ptr_cpu_f32, -, float)
ZENU_CPU_SCALAR_OP(zenu_compute_sub_mat_scalar_ptr_cpu_f64, -, double)
ZenuStatus zenu_compute_sub_mat_scalar_ptr_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_sub_mat_scalar_ptr_cpu_f32(
            dst, src, stride_dst, stride_src, scalar_ptr, n, f32);
    } else {
        return zenu_compute_sub_mat_scalar_ptr_cpu_f64(
            dst, src, stride_dst, stride_src, scalar_ptr, n, f64);
    }
}

ZENU_CPU_ASSIGN_OP(zenu_compute_sub_mat_mat_assign_cpu_f32, -=, float)
ZENU_CPU_ASSIGN_OP(zenu_compute_sub_mat_mat_assign_cpu_f64, -=, double)
ZenuStatus zenu_compute_sub_mat_mat_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_sub_mat_mat_assign_cpu_f32(
            dst, src, stride_dst, stride_src, n, f32);
    } else {
        return zenu_compute_sub_mat_mat_assign_cpu_f64(
            dst, src, stride_dst, stride_src, n, f64);
    }
}

ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_sub_mat_scalar_ptr_assign_cpu_f32, -=, float)
ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_sub_mat_scalar_ptr_assign_cpu_f64, -=, double)
ZenuStatus zenu_compute_sub_mat_scalar_ptr_assign_cpu(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_sub_mat_scalar_ptr_assign_cpu_f32(
            dst, stride_dst, scalar_ptr, n, f32);
    } else {
        return zenu_compute_sub_mat_scalar_ptr_assign_cpu_f64(
            dst, stride_dst, scalar_ptr, n, f64);
    }
}

/*=========================================================
 * MUL (CPU)
 *========================================================*/
ZENU_CPU_BINARY_OP(zenu_compute_mul_mat_mat_cpu_f32, *, float)
ZENU_CPU_BINARY_OP(zenu_compute_mul_mat_mat_cpu_f64, *, double)
ZenuStatus zenu_compute_mul_mat_mat_cpu(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_mul_mat_mat_cpu_f32(
            dst, src1, src2, stride_dst, stride_src1, stride_src2, n, f32);
    } else {
        return zenu_compute_mul_mat_mat_cpu_f64(
            dst, src1, src2, stride_dst, stride_src1, stride_src2, n, f64);
    }
}

ZENU_CPU_SCALAR_OP(zenu_compute_mul_mat_scalar_ptr_cpu_f32, *, float)
ZENU_CPU_SCALAR_OP(zenu_compute_mul_mat_scalar_ptr_cpu_f64, *, double)
ZenuStatus zenu_compute_mul_mat_scalar_ptr_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_mul_mat_scalar_ptr_cpu_f32(
            dst, src, stride_dst, stride_src, scalar_ptr, n, f32);
    } else {
        return zenu_compute_mul_mat_scalar_ptr_cpu_f64(
            dst, src, stride_dst, stride_src, scalar_ptr, n, f64);
    }
}

ZENU_CPU_ASSIGN_OP(zenu_compute_mul_mat_mat_assign_cpu_f32, *=, float)
ZENU_CPU_ASSIGN_OP(zenu_compute_mul_mat_mat_assign_cpu_f64, *=, double)
ZenuStatus zenu_compute_mul_mat_mat_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_mul_mat_mat_assign_cpu_f32(
            dst, src, stride_dst, stride_src, n, f32);
    } else {
        return zenu_compute_mul_mat_mat_assign_cpu_f64(
            dst, src, stride_dst, stride_src, n, f64);
    }
}

ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_mul_mat_scalar_ptr_assign_cpu_f32, *=, float)
ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_mul_mat_scalar_ptr_assign_cpu_f64, *=, double)
ZenuStatus zenu_compute_mul_mat_scalar_ptr_assign_cpu(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_mul_mat_scalar_ptr_assign_cpu_f32(
            dst, stride_dst, scalar_ptr, n, f32);
    } else {
        return zenu_compute_mul_mat_scalar_ptr_assign_cpu_f64(
            dst, stride_dst, scalar_ptr, n, f64);
    }
}

/*=========================================================
 * DIV (CPU)
 *========================================================*/
ZENU_CPU_BINARY_OP(zenu_compute_div_mat_mat_cpu_f32, /, float)
ZENU_CPU_BINARY_OP(zenu_compute_div_mat_mat_cpu_f64, /, double)
ZenuStatus zenu_compute_div_mat_mat_cpu(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_div_mat_mat_cpu_f32(
            dst, src1, src2, stride_dst, stride_src1, stride_src2, n, f32);
    } else {
        return zenu_compute_div_mat_mat_cpu_f64(
            dst, src1, src2, stride_dst, stride_src1, stride_src2, n, f64);
    }
}

ZENU_CPU_SCALAR_OP(zenu_compute_div_mat_scalar_ptr_cpu_f32, /, float)
ZENU_CPU_SCALAR_OP(zenu_compute_div_mat_scalar_ptr_cpu_f64, /, double)
ZenuStatus zenu_compute_div_mat_scalar_ptr_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_div_mat_scalar_ptr_cpu_f32(
            dst, src, stride_dst, stride_src, scalar_ptr, n, f32);
    } else {
        return zenu_compute_div_mat_scalar_ptr_cpu_f64(
            dst, src, stride_dst, stride_src, scalar_ptr, n, f64);
    }
}

ZENU_CPU_ASSIGN_OP(zenu_compute_div_mat_mat_assign_cpu_f32, /=, float)
ZENU_CPU_ASSIGN_OP(zenu_compute_div_mat_mat_assign_cpu_f64, /=, double)
ZenuStatus zenu_compute_div_mat_mat_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_div_mat_mat_assign_cpu_f32(
            dst, src, stride_dst, stride_src, n, f32);
    } else {
        return zenu_compute_div_mat_mat_assign_cpu_f64(
            dst, src, stride_dst, stride_src, n, f64);
    }
}

ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_div_mat_scalar_ptr_assign_cpu_f32, /=, float)
ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_div_mat_scalar_ptr_assign_cpu_f64, /=, double)
ZenuStatus zenu_compute_div_mat_scalar_ptr_assign_cpu(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type)
{
    if (data_type == f32) {
        return zenu_compute_div_mat_scalar_ptr_assign_cpu_f32(
            dst, stride_dst, scalar_ptr, n, f32);
    } else {
        return zenu_compute_div_mat_scalar_ptr_assign_cpu_f64(
            dst, stride_dst, scalar_ptr, n, f64);
    }
}

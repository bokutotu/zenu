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
#include "utils.h"
#include "iter_macro.h"

#include <omp.h>
#include <string.h>
#include <stdio.h>

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

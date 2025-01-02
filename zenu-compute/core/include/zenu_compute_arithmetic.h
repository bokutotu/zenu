#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file zenu_arith.h
 * @brief Arithmetic functions (add/sub/mul/div) on CPU / "nvidia" GPU, with detailed docs for each function.
 *
 * Each operation has 6 forms:
 *  1) mat_mat
 *  2) mat_scalar
 *  3) mat_scalar_ptr
 *  4) mat_mat_assign
 *  5) mat_scalar_assign
 *  6) mat_scalar_ptr_assign
 *
 * For each form, we provide:
 *  - A CPU version (suffix `_cpu`)
 *  - A "nvidia" version (suffix `_nvidia`)
 *
 * Both versions include:
 *  - `n` (number of elements)
 *  - `ZenuDataType data_type`
 *
 * The "nvidia" version does NOT require a device_id, per requirement.
 */

#include <stddef.h> // for size_t

/*======================================================================
 *                        ADD  (mat + ...)
 *=====================================================================*/

/*------------------ 1) ADD: mat + mat ------------------*/
/**
 * @brief Add two matrices (CPU): dst[i] = src1[i] + src2[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src1        CPU memory pointer to first operand
 * @param[in]     src2        CPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements to process
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_add_mat_mat_cpu(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Add two matrices ("nvidia"): dst[i] = src1[i] + src2[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src1        GPU memory pointer to first operand
 * @param[in]     src2        GPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_add_mat_mat_nvidia(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) ADD: mat + scalar ------------------*/
/**
 * @brief Add a scalar to each element of a matrix (CPU).
 *        dst[i] = src[i] + scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to add
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_add_mat_scalar_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Add a scalar to each element of a matrix ("nvidia").
 *        dst[i] = src[i] + scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to add
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_add_mat_scalar_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 3) ADD: mat + *(scalar_ptr) ------------------*/
/**
 * @brief Add a pointer-based scalar to each element (CPU).
 *        dst[i] = src[i] + (*(scalar_ptr))
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_add_mat_scalar_ptr_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Add a pointer-based scalar to each element ("nvidia").
 *        dst[i] = src[i] + (*(scalar_ptr))
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_add_mat_scalar_ptr_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 4) ADD: mat_mat_assign (dst += src) ------------------*/
/**
 * @brief Add assignment (CPU): dst[i] += src[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_add_mat_mat_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Add assignment ("nvidia"): dst[i] += src[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_add_mat_mat_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 5) ADD: mat_scalar_assign (dst += scalar) ------------------*/
/**
 * @brief Add scalar assignment (CPU): dst[i] += scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to add
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_add_mat_scalar_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Add scalar assignment ("nvidia"): dst[i] += scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to add
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_add_mat_scalar_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 6) ADD: mat_scalar_ptr_assign (dst += *(scalar_ptr)) ------------------*/
/**
 * @brief Add scalar_ptr assignment (CPU): dst[i] += *(scalar_ptr)
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_add_mat_scalar_ptr_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Add scalar_ptr assignment ("nvidia"): dst[i] += *(scalar_ptr)
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_add_mat_scalar_ptr_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                        SUB  (mat - ...)
 *=====================================================================*/

/*------------------ 1) SUB: mat - mat ------------------*/
/**
 * @brief Subtract two matrices (CPU): dst[i] = src1[i] - src2[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src1        CPU memory pointer to first operand
 * @param[in]     src2        CPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements to process
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_sub_mat_mat_cpu(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Subtract two matrices ("nvidia"): dst[i] = src1[i] - src2[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src1        GPU memory pointer to first operand
 * @param[in]     src2        GPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_sub_mat_mat_nvidia(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) SUB: mat - scalar ------------------*/
/**
 * @brief Subtract a scalar from each element of a matrix (CPU).
 *        dst[i] = src[i] - scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to subtract
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_sub_mat_scalar_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Subtract a scalar from each element of a matrix ("nvidia").
 *        dst[i] = src[i] - scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to subtract
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_sub_mat_scalar_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 3) SUB: mat - *(scalar_ptr) ------------------*/
/**
 * @brief Subtract a pointer-based scalar from each element (CPU).
 *        dst[i] = src[i] - (*(scalar_ptr))
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_sub_mat_scalar_ptr_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Subtract a pointer-based scalar from each element ("nvidia").
 *        dst[i] = src[i] - (*(scalar_ptr))
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_sub_mat_scalar_ptr_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 4) SUB: mat_mat_assign (dst -= src) ------------------*/
/**
 * @brief Subtract assignment (CPU): dst[i] -= src[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_sub_mat_mat_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Subtract assignment ("nvidia"): dst[i] -= src[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_sub_mat_mat_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 5) SUB: mat_scalar_assign (dst -= scalar) ------------------*/
/**
 * @brief Subtract scalar assignment (CPU): dst[i] -= scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to subtract
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_sub_mat_scalar_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Subtract scalar assignment ("nvidia"): dst[i] -= scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to subtract
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_sub_mat_scalar_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 6) SUB: mat_scalar_ptr_assign (dst -= *(scalar_ptr)) ------------------*/
/**
 * @brief Subtract scalar_ptr assignment (CPU): dst[i] -= *(scalar_ptr)
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_sub_mat_scalar_ptr_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Subtract scalar_ptr assignment ("nvidia"): dst[i] -= *(scalar_ptr)
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_sub_mat_scalar_ptr_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                        MUL  (mat * ...)
 *=====================================================================*/

/*------------------ 1) MUL: mat * mat ------------------*/
/**
 * @brief Multiply two matrices (CPU): dst[i] = src1[i] * src2[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src1        CPU memory pointer to first operand
 * @param[in]     src2        CPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements to process
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_mul_mat_mat_cpu(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply two matrices ("nvidia"): dst[i] = src1[i] * src2[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src1        GPU memory pointer to first operand
 * @param[in]     src2        GPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_mul_mat_mat_nvidia(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) MUL: mat * scalar ------------------*/
/**
 * @brief Multiply a matrix by a scalar (CPU): dst[i] = src[i] * scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to multiply
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply a matrix by a scalar ("nvidia"): dst[i] = src[i] * scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to multiply
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 3) MUL: mat * *(scalar_ptr) ------------------*/
/**
 * @brief Multiply a pointer-based scalar with each element (CPU):
 *        dst[i] = src[i] * (*(scalar_ptr))
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_ptr_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply a pointer-based scalar with each element ("nvidia"):
 *        dst[i] = src[i] * (*(scalar_ptr))
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_ptr_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 4) MUL: mat_mat_assign (dst *= src) ------------------*/
/**
 * @brief Multiply assignment (CPU): dst[i] *= src[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_mat_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply assignment ("nvidia"): dst[i] *= src[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_mat_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 5) MUL: mat_scalar_assign (dst *= scalar) ------------------*/
/**
 * @brief Multiply scalar assignment (CPU): dst[i] *= scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to multiply
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply scalar assignment ("nvidia"): dst[i] *= scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to multiply
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 6) MUL: mat_scalar_ptr_assign (dst *= *(scalar_ptr)) ------------------*/
/**
 * @brief Multiply scalar_ptr assignment (CPU): dst[i] *= *(scalar_ptr)
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_ptr_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply scalar_ptr assignment ("nvidia"): dst[i] *= *(scalar_ptr)
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_ptr_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                        DIV  (mat / ...)
 *=====================================================================*/

/*------------------ 1) DIV: mat / mat ------------------*/
/**
 * @brief Divide two matrices (CPU): dst[i] = src1[i] / src2[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src1        CPU memory pointer to first operand
 * @param[in]     src2        CPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements to process
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_div_mat_mat_cpu(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide two matrices ("nvidia"): dst[i] = src1[i] / src2[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src1        GPU memory pointer to first operand
 * @param[in]     src2        GPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_div_mat_mat_nvidia(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) DIV: mat / scalar ------------------*/
/**
 * @brief Divide a matrix by a scalar (CPU): dst[i] = src[i] / scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to divide by
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide a matrix by a scalar ("nvidia"): dst[i] = src[i] / scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to divide by
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 3) DIV: mat / *(scalar_ptr) ------------------*/
/**
 * @brief Divide each element of a matrix by a pointer-based scalar (CPU):
 *        dst[i] = src[i] / (*(scalar_ptr))
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_ptr_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide each element of a matrix by a pointer-based scalar ("nvidia"):
 *        dst[i] = src[i] / (*(scalar_ptr))
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_ptr_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 4) DIV: mat_mat_assign (dst /= src) ------------------*/
/**
 * @brief Divide assignment (CPU): dst[i] /= src[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_mat_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide assignment ("nvidia"): dst[i] /= src[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_mat_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 5) DIV: mat_scalar_assign (dst /= scalar) ------------------*/
/**
 * @brief Divide scalar assignment (CPU): dst[i] /= scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to divide by
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide scalar assignment ("nvidia"): dst[i] /= scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to divide by
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 6) DIV: mat_scalar_ptr_assign (dst /= *(scalar_ptr)) ------------------*/
/**
 * @brief Divide scalar_ptr assignment (CPU): dst[i] /= *(scalar_ptr)
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_ptr_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide scalar_ptr assignment ("nvidia"): dst[i] /= *(scalar_ptr)
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_ptr_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                        MUL  (mat * ...)
 *=====================================================================*/

/*------------------ 1) MUL: mat * mat ------------------*/
/**
 * @brief Multiply two matrices (CPU): dst[i] = src1[i] * src2[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src1        CPU memory pointer to first operand
 * @param[in]     src2        CPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements to process
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_mul_mat_mat_cpu(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply two matrices ("nvidia"): dst[i] = src1[i] * src2[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src1        GPU memory pointer to first operand
 * @param[in]     src2        GPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_mul_mat_mat_nvidia(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) MUL: mat * scalar ------------------*/
/**
 * @brief Multiply a matrix by a scalar (CPU): dst[i] = src[i] * scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to multiply
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply a matrix by a scalar ("nvidia"): dst[i] = src[i] * scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to multiply
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 3) MUL: mat * *(scalar_ptr) ------------------*/
/**
 * @brief Multiply a pointer-based scalar with each element (CPU):
 *        dst[i] = src[i] * (*(scalar_ptr))
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_ptr_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply a pointer-based scalar with each element ("nvidia"):
 *        dst[i] = src[i] * (*(scalar_ptr))
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_ptr_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 4) MUL: mat_mat_assign (dst *= src) ------------------*/
/**
 * @brief Multiply assignment (CPU): dst[i] *= src[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_mat_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply assignment ("nvidia"): dst[i] *= src[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_mat_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 5) MUL: mat_scalar_assign (dst *= scalar) ------------------*/
/**
 * @brief Multiply scalar assignment (CPU): dst[i] *= scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to multiply
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply scalar assignment ("nvidia"): dst[i] *= scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to multiply
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 6) MUL: mat_scalar_ptr_assign (dst *= *(scalar_ptr)) ------------------*/
/**
 * @brief Multiply scalar_ptr assignment (CPU): dst[i] *= *(scalar_ptr)
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_ptr_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Multiply scalar_ptr assignment ("nvidia"): dst[i] *= *(scalar_ptr)
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_mul_mat_scalar_ptr_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                        DIV  (mat / ...)
 *=====================================================================*/

/*------------------ 1) DIV: mat / mat ------------------*/
/**
 * @brief Divide two matrices (CPU): dst[i] = src1[i] / src2[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src1        CPU memory pointer to first operand
 * @param[in]     src2        CPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements to process
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_div_mat_mat_cpu(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide two matrices ("nvidia"): dst[i] = src1[i] / src2[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src1        GPU memory pointer to first operand
 * @param[in]     src2        GPU memory pointer to second operand
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src1 Stride for src1
 * @param[in]     stride_src2 Stride for src2
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus         Success or error code
 */
ZenuStatus zenu_compute_div_mat_mat_nvidia(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) DIV: mat / scalar ------------------*/
/**
 * @brief Divide a matrix by a scalar (CPU): dst[i] = src[i] / scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to divide by
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide a matrix by a scalar ("nvidia"): dst[i] = src[i] / scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to divide by
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 3) DIV: mat / *(scalar_ptr) ------------------*/
/**
 * @brief Divide each element of a matrix by a pointer-based scalar (CPU):
 *        dst[i] = src[i] / (*(scalar_ptr))
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_ptr_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide each element of a matrix by a pointer-based scalar ("nvidia"):
 *        dst[i] = src[i] / (*(scalar_ptr))
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_ptr_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 4) DIV: mat_mat_assign (dst /= src) ------------------*/
/**
 * @brief Divide assignment (CPU): dst[i] /= src[i]
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_mat_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide assignment ("nvidia"): dst[i] /= src[i]
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_mat_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 5) DIV: mat_scalar_assign (dst /= scalar) ------------------*/
/**
 * @brief Divide scalar assignment (CPU): dst[i] /= scalar
 *
 * @param[in,out] dst        CPU memory pointer to output buffer
 * @param[in]     src        CPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to divide by
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide scalar assignment ("nvidia"): dst[i] /= scalar
 *
 * @param[in,out] dst        GPU memory pointer to output buffer
 * @param[in]     src        GPU memory pointer to input matrix
 * @param[in]     stride_dst Stride for dst
 * @param[in]     stride_src Stride for src
 * @param[in]     scalar     The scalar value to divide by
 * @param[in]     n          Number of elements
 * @param[in]     data_type  f32 or f64
 * @return ZenuStatus        Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    float       scalar,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 6) DIV: mat_scalar_ptr_assign (dst /= *(scalar_ptr)) ------------------*/
/**
 * @brief Divide scalar_ptr assignment (CPU): dst[i] /= *(scalar_ptr)
 *
 * @param[in,out] dst         CPU memory pointer to output buffer
 * @param[in]     src         CPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_ptr_assign_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief Divide scalar_ptr assignment ("nvidia"): dst[i] /= *(scalar_ptr)
 *
 * @param[in,out] dst         GPU memory pointer to output buffer
 * @param[in]     src         GPU memory pointer to input matrix
 * @param[in]     stride_dst  Stride for dst
 * @param[in]     stride_src   Stride for src
 * @param[in]     scalar_ptr  Pointer to scalar (float* or double*)
 * @param[in]     n           Number of elements
 * @param[in]     data_type   f32 or f64
 * @return ZenuStatus          Success or error code
 */
ZenuStatus zenu_compute_div_mat_scalar_ptr_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

#ifdef __cplusplus
}
#endif


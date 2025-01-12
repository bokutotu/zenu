#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file zenu_compute_arithmetic.h
 * @brief 四則演算 (加減乗除) を行う関数群 (CPU / "nvidia" GPU 対応)。
 *
 * 本ヘッダファイルでは、zenu_compute.h に定義された ZenuDataType や ZenuStatus を用いて
 * CPU / "nvidia" GPU 上での加減乗除を行う関数を提供します。
 *
 * 各演算 (add, sub, mul, div) は、下記のような形で定義されています。
 *  - mat_mat:         dst = src1 (+,-,*,/) src2
 *  - mat_scalar:      dst = src (+,-,*,/) scalar        (今回は例示のみ、コード省略)
 *  - mat_scalar_ptr:  dst = src (+,-,*,/) *scalar_ptr
 *  - mat_mat_assign:  dst += src  (または -=, *=, /=)
 *  - mat_scalar_assign:     dst += scalar       (今回は例示のみ、コード省略)
 *  - mat_scalar_ptr_assign: dst += *scalar_ptr
 *
 * それぞれ CPU 用 (関数名末尾 `_cpu`) と "nvidia" GPU 用 (関数名末尾 `_nvidia`) が存在します。
 */

#include <stddef.h>
#include "zenu_compute_type.h"

/*======================================================================
 *                        ADD  (mat + ...)
 *=====================================================================*/

/*------------------ 1) ADD: mat + mat ------------------*/
/**
 * @brief 2つの配列要素を加算 (CPU): dst[i] = src1[i] + src2[i]
 *
 * @param[in,out] dst         CPU メモリ上の出力バッファへのポインタ
 * @param[in]     src1        CPU メモリ上の第1オペランド
 * @param[in]     src2        CPU メモリ上の第2オペランド
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     stride_src1 src1 のストライド (要素間隔)
 * @param[in]     stride_src2 src2 のストライド (要素間隔)
 * @param[in]     n           処理する要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 2つの配列要素を加算 ("nvidia"): dst[i] = src1[i] + src2[i]
 *
 * @param[in,out] dst         GPU メモリ上の出力バッファへのポインタ
 * @param[in]     src1        GPU メモリ上の第1オペランド
 * @param[in]     src2        GPU メモリ上の第2オペランド
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     stride_src1 src1 のストライド (要素間隔)
 * @param[in]     stride_src2 src2 のストライド (要素間隔)
 * @param[in]     n           処理する要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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

/*------------------ 3) ADD: mat + *(scalar_ptr) ------------------*/
/**
 * @brief スカラー(ポインタ)を加算 (CPU): dst[i] = src[i] + (*(scalar_ptr))
 *
 * @param[in,out] dst         CPU メモリ上の出力バッファへのポインタ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     stride_src  src のストライド (要素間隔)
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           処理する要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief スカラー(ポインタ)を加算 ("nvidia"): dst[i] = src[i] + (*(scalar_ptr))
 *
 * @param[in,out] dst         GPU メモリ上の出力バッファへのポインタ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     stride_src  src のストライド (要素間隔)
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           処理する要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 加算代入 (CPU): dst[i] += src[i]
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     stride_src  src のストライド (要素間隔)
 * @param[in]     n           処理する要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 加算代入 ("nvidia"): dst[i] += src[i]
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     stride_src  src のストライド (要素間隔)
 * @param[in]     n           処理する要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_add_mat_mat_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 6) ADD: mat_scalar_ptr_assign (dst += *(scalar_ptr)) ------------------*/
/**
 * @brief スカラー(ポインタ)の加算代入 (CPU): dst[i] += *(scalar_ptr)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           処理する要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_add_mat_scalar_ptr_assign_cpu(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief スカラー(ポインタ)の加算代入 ("nvidia"): dst[i] += *(scalar_ptr)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           処理する要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_add_mat_scalar_ptr_assign_nvidia(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                        SUB  (mat - ...)
 *=====================================================================*/

/*------------------ 1) SUB: mat - mat ------------------*/
/**
 * @brief 2つの配列要素を減算 (CPU): dst[i] = src1[i] - src2[i]
 *
 * @param[in,out] dst         CPU メモリ上の出力バッファ
 * @param[in]     src1        CPU メモリ上の第1オペランド
 * @param[in]     src2        CPU メモリ上の第2オペランド
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src1 src1 のストライド
 * @param[in]     stride_src2 src2 のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 2つの配列要素を減算 ("nvidia"): dst[i] = src1[i] - src2[i]
 *
 * @param[in,out] dst         GPU メモリ上の出力バッファ
 * @param[in]     src1        GPU メモリ上の第1オペランド
 * @param[in]     src2        GPU メモリ上の第2オペランド
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src1 src1 のストライド
 * @param[in]     stride_src2 src2 のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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

/*------------------ 3) SUB: mat - *(scalar_ptr) ------------------*/
/**
 * @brief スカラー(ポインタ)の減算 (CPU): dst[i] = src[i] - (*(scalar_ptr))
 *
 * @param[in,out] dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief スカラー(ポインタ)の減算 ("nvidia"): dst[i] = src[i] - (*(scalar_ptr))
 *
 * @param[in,out] dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 減算代入 (CPU): dst[i] -= src[i]
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 減算代入 ("nvidia"): dst[i] -= src[i]
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_sub_mat_mat_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 6) SUB: mat_scalar_ptr_assign (dst -= *(scalar_ptr)) ------------------*/
/**
 * @brief スカラー(ポインタ)の減算代入 (CPU): dst[i] -= *(scalar_ptr)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_sub_mat_scalar_ptr_assign_cpu(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief スカラー(ポインタ)の減算代入 ("nvidia"): dst[i] -= *(scalar_ptr)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_sub_mat_scalar_ptr_assign_nvidia(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                        MUL  (mat * ...)
 *=====================================================================*/

/*------------------ 1) MUL: mat * mat ------------------*/
/**
 * @brief 2つの配列要素を乗算 (CPU): dst[i] = src1[i] * src2[i]
 *
 * @param[in,out] dst         CPU メモリ上の出力バッファ
 * @param[in]     src1        CPU メモリ上の第1オペランド
 * @param[in]     src2        CPU メモリ上の第2オペランド
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src1 src1 のストライド
 * @param[in]     stride_src2 src2 のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 2つの配列要素を乗算 ("nvidia"): dst[i] = src1[i] * src2[i]
 *
 * @param[in,out] dst         GPU メモリ上の出力バッファ
 * @param[in]     src1        GPU メモリ上の第1オペランド
 * @param[in]     src2        GPU メモリ上の第2オペランド
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src1 src1 のストライド
 * @param[in]     stride_src2 src2 のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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

/*------------------ 3) MUL: mat * *(scalar_ptr) ------------------*/
/**
 * @brief スカラー(ポインタ)との乗算 (CPU): dst[i] = src[i] * (*(scalar_ptr))
 *
 * @param[in,out] dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief スカラー(ポインタ)との乗算 ("nvidia"): dst[i] = src[i] * (*(scalar_ptr))
 *
 * @param[in,out] dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 乗算代入 (CPU): dst[i] *= src[i]
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 乗算代入 ("nvidia"): dst[i] *= src[i]
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_mul_mat_mat_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 6) MUL: mat_scalar_ptr_assign (dst *= *(scalar_ptr)) ------------------*/
/**
 * @brief スカラー(ポインタ)の乗算代入 (CPU): dst[i] *= *(scalar_ptr)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_mul_mat_scalar_ptr_assign_cpu(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief スカラー(ポインタ)の乗算代入 ("nvidia"): dst[i] *= *(scalar_ptr)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_mul_mat_scalar_ptr_assign_nvidia(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                        DIV  (mat / ...)
 *=====================================================================*/

/*------------------ 1) DIV: mat / mat ------------------*/
/**
 * @brief 2つの配列要素を除算 (CPU): dst[i] = src1[i] / src2[i]
 *
 * @param[in,out] dst         CPU メモリ上の出力バッファ
 * @param[in]     src1        CPU メモリ上の第1オペランド
 * @param[in]     src2        CPU メモリ上の第2オペランド
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src1 src1 のストライド
 * @param[in]     stride_src2 src2 のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 2つの配列要素を除算 ("nvidia"): dst[i] = src1[i] / src2[i]
 *
 * @param[in,out] dst         GPU メモリ上の出力バッファ
 * @param[in]     src1        GPU メモリ上の第1オペランド
 * @param[in]     src2        GPU メモリ上の第2オペランド
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src1 src1 のストライド
 * @param[in]     stride_src2 src2 のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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

/*------------------ 3) DIV: mat / *(scalar_ptr) ------------------*/
/**
 * @brief スカラー(ポインタ)による除算 (CPU): dst[i] = src[i] / (*(scalar_ptr))
 *
 * @param[in,out] dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief スカラー(ポインタ)による除算 ("nvidia"): dst[i] = src[i] / (*(scalar_ptr))
 *
 * @param[in,out] dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 除算代入 (CPU): dst[i] /= src[i]
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
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
 * @brief 除算代入 ("nvidia"): dst[i] /= src[i]
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_div_mat_mat_assign_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 6) DIV: mat_scalar_ptr_assign (dst /= *(scalar_ptr)) ------------------*/
/**
 * @brief スカラー(ポインタ)の除算代入 (CPU): dst[i] /= *(scalar_ptr)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_div_mat_scalar_ptr_assign_cpu(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief スカラー(ポインタ)の除算代入 ("nvidia"): dst[i] /= *(scalar_ptr)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     scalar_ptr  スカラー (float* または double*) へのポインタ
 * @param[in]     n           要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_div_mat_scalar_ptr_assign_nvidia(
    void*       dst,
    int         stride_dst,
    const void* scalar_ptr,
    size_t      n,
    ZenuDataType data_type
);

#ifdef __cplusplus
}
#endif


#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file zenu_compute_trigonometric.h
 * @brief 三角関数・双曲線関数 (sin, cos, tan, sinh, cosh, tanh) を行う関数群。
 *
 * 本ヘッダファイルでは、zenu_compute.h に定義された ZenuDataType や ZenuStatus を用いて
 * CPU / "nvidia" GPU 上で各種三角関数・双曲線関数を計算します。
 *
 * 関数は大きく2種類存在します:
 *  - out-of-place:   dst = f(src)  
 *    （2つの配列を用意し、入力 src に対して演算を行い、その結果を出力 dst に格納）
 *  - in-place:       dst = f(dst)  
 *    （単一の配列を用意し、そのまま上書きして結果を得る）
 *
 * それぞれ CPU 用 (関数名末尾 `_cpu`) と "nvidia" GPU 用 (関数名末尾 `_nvidia`) があります。
 */

#include <stddef.h>
#include "zenu_compute_type.h"

/*======================================================================
 *                     SIN
 *=====================================================================*/

/*------------------ 1) out-of-place: dst = sin(src) ------------------*/

/**
 * @brief 配列 src の各要素に対して正弦 (sin) を適用し、その結果を dst に格納 (CPU版)
 *
 * @param[out]    dst         CPU メモリ上の出力バッファ（演算結果の格納先）
 * @param[in]     src         CPU メモリ上の入力バッファ（演算対象）
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     stride_src  src のストライド (要素間隔)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64 の指定
 * @return ZenuStatus         成功した場合は ZENU_SUCCESS、エラー時は適切なエラーコードを返す
 */
ZenuStatus zenu_compute_sin_mat_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対して正弦 (sin) を適用し、その結果を dst に格納 ("nvidia"版)
 *
 * @param[out]    dst         GPU メモリ上の出力バッファ（演算結果の格納先）
 * @param[in]     src         GPU メモリ上の入力バッファ（演算対象）
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     stride_src  src のストライド (要素間隔)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64 の指定
 * @return ZenuStatus         成功した場合は ZENU_SUCCESS、エラー時は適切なエラーコードを返す
 */
ZenuStatus zenu_compute_sin_mat_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) in-place: dst = sin(dst) ------------------*/

/**
 * @brief 配列 dst の各要素に対して正弦 (sin) を適用し、そのまま上書きする (CPU版)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ（演算対象 & 演算結果の格納先）
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64 の指定
 * @return ZenuStatus         成功した場合は ZENU_SUCCESS、エラー時は適切なエラーコードを返す
 */
ZenuStatus zenu_compute_sin_mat_assign_cpu(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して正弦 (sin) を適用し、そのまま上書きする ("nvidia"版)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ（演算対象 & 演算結果の格納先）
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64 の指定
 * @return ZenuStatus         成功した場合は ZENU_SUCCESS、エラー時は適切なエラーコードを返す
 */
ZenuStatus zenu_compute_sin_mat_assign_nvidia(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                     COS
 *=====================================================================*/

/*------------------ 1) out-of-place: dst = cos(src) ------------------*/

/**
 * @brief 配列 src の各要素に対して余弦 (cos) を適用し、その結果を dst に格納 (CPU版)
 *
 * @param[out]    dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_cos_mat_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対して余弦 (cos) を適用し、その結果を dst に格納 ("nvidia"版)
 *
 * @param[out]    dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_cos_mat_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) in-place: dst = cos(dst) ------------------*/

/**
 * @brief 配列 dst の各要素に対して余弦 (cos) を適用し、そのまま上書きする (CPU版)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_cos_mat_assign_cpu(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して余弦 (cos) を適用し、そのまま上書きする ("nvidia"版)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_cos_mat_assign_nvidia(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                     TAN
 *=====================================================================*/

/*------------------ 1) out-of-place: dst = tan(src) ------------------*/

/**
 * @brief 配列 src の各要素に対して正接 (tan) を適用し、その結果を dst に格納 (CPU版)
 *
 * @param[out]    dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_tan_mat_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対して正接 (tan) を適用し、その結果を dst に格納 ("nvidia"版)
 *
 * @param[out]    dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_tan_mat_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) in-place: dst = tan(dst) ------------------*/

/**
 * @brief 配列 dst の各要素に対して正接 (tan) を適用し、そのまま上書きする (CPU版)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_tan_mat_assign_cpu(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して正接 (tan) を適用し、そのまま上書きする ("nvidia"版)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_tan_mat_assign_nvidia(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                     SINH
 *=====================================================================*/

/*------------------ 1) out-of-place: dst = sinh(src) ------------------*/

/**
 * @brief 配列 src の各要素に対して双曲線正弦 (sinh) を適用し、その結果を dst に格納 (CPU版)
 *
 * @param[out]    dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_sinh_mat_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対して双曲線正弦 (sinh) を適用し、その結果を dst に格納 ("nvidia"版)
 *
 * @param[out]    dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_sinh_mat_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) in-place: dst = sinh(dst) ------------------*/

/**
 * @brief 配列 dst の各要素に対して双曲線正弦 (sinh) を適用し、そのまま上書きする (CPU版)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_sinh_mat_assign_cpu(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して双曲線正弦 (sinh) を適用し、そのまま上書きする ("nvidia"版)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_sinh_mat_assign_nvidia(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                     COSH
 *=====================================================================*/

/*------------------ 1) out-of-place: dst = cosh(src) ------------------*/

/**
 * @brief 配列 src の各要素に対して双曲線余弦 (cosh) を適用し、その結果を dst に格納 (CPU版)
 *
 * @param[out]    dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_cosh_mat_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対して双曲線余弦 (cosh) を適用し、その結果を dst に格納 ("nvidia"版)
 *
 * @param[out]    dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_cosh_mat_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) in-place: dst = cosh(dst) ------------------*/

/**
 * @brief 配列 dst の各要素に対して双曲線余弦 (cosh) を適用し、そのまま上書きする (CPU版)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_cosh_mat_assign_cpu(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して双曲線余弦 (cosh) を適用し、そのまま上書きする ("nvidia"版)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_cosh_mat_assign_nvidia(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/*======================================================================
 *                     TANH
 *=====================================================================*/

/*------------------ 1) out-of-place: dst = tanh(src) ------------------*/

/**
 * @brief 配列 src の各要素に対して双曲線正接 (tanh) を適用し、その結果を dst に格納 (CPU版)
 *
 * @param[out]    dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_tanh_mat_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対して双曲線正接 (tanh) を適用し、その結果を dst に格納 ("nvidia"版)
 *
 * @param[out]    dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_tanh_mat_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/*------------------ 2) in-place: dst = tanh(dst) ------------------*/

/**
 * @brief 配列 dst の各要素に対して双曲線正接 (tanh) を適用し、そのまま上書きする (CPU版)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_tanh_mat_assign_cpu(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して双曲線正接 (tanh) を適用し、そのまま上書きする ("nvidia"版)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_tanh_mat_assign_nvidia(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

#ifdef __cplusplus
}
#endif


#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file zenu_compute_extras.h
 * @brief 指数関数 (exp), 自然対数 (ln), 絶対値 (abs), クリップ (clip), べき乗 (pow) を
 *        行う関数群。
 *
 * 本ヘッダファイルでは、zenu_compute.h に定義された ZenuDataType や ZenuStatus を用いて
 * CPU / "nvidia" GPU 上で各種演算 (exp, ln, abs, clip, pow) を計算します。
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

/**
 * @brief 配列 src の各要素に対して指数関数 (exp) を適用し、その結果を dst に格納 (CPU版)
 *
 * @param[out]    dst         CPU メモリ上の出力バッファ（演算結果の格納先）
 * @param[in]     src         CPU メモリ上の入力バッファ（演算対象）
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     stride_src  src のストライド (要素間隔)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64 の指定
 * @return ZenuStatus         成功 (Success) またはエラーコード
 */
ZenuStatus zenu_compute_exp_mat_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対して指数関数 (exp) を適用し、その結果を dst に格納 ("nvidia"版)
 *
 * @param[out]    dst         GPU メモリ上の出力バッファ（演算結果の格納先）
 * @param[in]     src         GPU メモリ上の入力バッファ（演算対象）
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     stride_src  src のストライド (要素間隔)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64 の指定
 * @return ZenuStatus         成功 (Success) またはエラーコード
 */
ZenuStatus zenu_compute_exp_mat_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して指数関数 (exp) を適用し、そのまま上書きする (CPU版)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ（演算対象 & 演算結果の格納先）
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64 の指定
 * @return ZenuStatus         成功 (Success) またはエラーコード
 */
ZenuStatus zenu_compute_exp_mat_assign_cpu(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して指数関数 (exp) を適用し、そのまま上書きする ("nvidia"版)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ（演算対象 & 演算結果の格納先）
 * @param[in]     stride_dst  dst のストライド (要素間隔)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64 の指定
 * @return ZenuStatus         成功 (Success) またはエラーコード
 */
ZenuStatus zenu_compute_exp_mat_assign_nvidia(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対して自然対数 (ln) を適用し、その結果を dst に格納 (CPU版)
 *
 * @param[out]    dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_ln_mat_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対して自然対数 (ln) を適用し、その結果を dst に格納 ("nvidia"版)
 *
 * @param[out]    dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_ln_mat_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して自然対数 (ln) を適用し、そのまま上書きする (CPU版)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_ln_mat_assign_cpu(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して自然対数 (ln) を適用し、そのまま上書きする ("nvidia"版)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_ln_mat_assign_nvidia(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対して絶対値 (abs) を適用し、その結果を dst に格納 (CPU版)
 *
 * @param[out]    dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_abs_mat_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対して絶対値 (abs) を適用し、その結果を dst に格納 ("nvidia"版)
 *
 * @param[out]    dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_abs_mat_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して絶対値 (abs) を適用し、そのまま上書きする (CPU版)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_abs_mat_assign_cpu(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対して絶対値 (abs) を適用し、そのまま上書きする ("nvidia"版)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_abs_mat_assign_nvidia(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対し、クリップ (clip) 処理を行い、その結果を dst に格納 (CPU版)
 *
 *        clip 処理: 
 *          - 要素 x が min_val より小さい場合は min_val として扱う
 *          - 要素 x が max_val より大きい場合は max_val として扱う
 *          - それ以外は x をそのまま使用
 *
 * @param[out]    dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @param[in]     min_val     クリップの下限値 (double で指定、data_type に応じて変換)
 * @param[in]     max_val     クリップの上限値 (double で指定、data_type に応じて変換)
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_clip_mat_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type,
    double      min_val,
    double      max_val
);

/**
 * @brief 配列 src の各要素に対し、クリップ (clip) 処理を行い、その結果を dst に格納 ("nvidia"版)
 *
 *        clip 処理: 
 *          - 要素 x が min_val より小さい場合は min_val として扱う
 *          - 要素 x が max_val より大きい場合は max_val として扱う
 *          - それ以外は x をそのまま使用
 *
 * @param[out]    dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @param[in]     min_val     クリップの下限値 (double で指定、data_type に応じて変換)
 * @param[in]     max_val     クリップの上限値 (double で指定、data_type に応じて変換)
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_clip_mat_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    size_t      n,
    ZenuDataType data_type,
    double      min_val,
    double      max_val
);

/**
 * @brief 配列 dst の各要素に対し、クリップ (clip) 処理を行い、そのまま上書きする (CPU版)
 *
 *        clip 処理: 
 *          - 要素 x が min_val より小さい場合は min_val として扱う
 *          - 要素 x が max_val より大きい場合は max_val として扱う
 *          - それ以外は x をそのまま使用
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @param[in]     min_val     クリップの下限値 (double で指定、data_type に応じて変換)
 * @param[in]     max_val     クリップの上限値 (double で指定、data_type に応じて変換)
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_clip_mat_assign_cpu(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type,
    double      min_val,
    double      max_val
);

/**
 * @brief 配列 dst の各要素に対し、クリップ (clip) 処理を行い、そのまま上書きする ("nvidia"版)
 *
 *        clip 処理: 
 *          - 要素 x が min_val より小さい場合は min_val として扱う
 *          - 要素 x が max_val より大きい場合は max_val として扱う
 *          - それ以外は x をそのまま使用
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @param[in]     min_val     クリップの下限値 (double で指定、data_type に応じて変換)
 * @param[in]     max_val     クリップの上限値 (double で指定、data_type に応じて変換)
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_clip_mat_assign_nvidia(
    void*       dst,
    int         stride_dst,
    size_t      n,
    ZenuDataType data_type,
    double      min_val,
    double      max_val
);

/**
 * @brief 配列 src の各要素に対してべき乗 (pow) を適用し、その結果を dst に格納 (CPU版)
 *
 * @param[out]    dst         CPU メモリ上の出力バッファ
 * @param[in]     src         CPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     exponent    べき乗の指数 (double で指定、data_type に応じて変換)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_pow_mat_cpu(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* exponent,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 src の各要素に対してべき乗 (pow) を適用し、その結果を dst に格納 ("nvidia"版)
 *
 * @param[out]    dst         GPU メモリ上の出力バッファ
 * @param[in]     src         GPU メモリ上の入力バッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     stride_src  src のストライド
 * @param[in]     exponent    べき乗の指数 (double で指定、data_type に応じて変換)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_pow_mat_nvidia(
    void*       dst,
    const void* src,
    int         stride_dst,
    int         stride_src,
    const void* exponent,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対してべき乗 (pow) を適用し、そのまま上書きする (CPU版)
 *
 * @param[in,out] dst         CPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     exponent    べき乗の指数 (double で指定、data_type に応じて変換)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_pow_mat_assign_cpu(
    void*       dst,
    int         stride_dst,
    const void* exponent,
    size_t      n,
    ZenuDataType data_type
);

/**
 * @brief 配列 dst の各要素に対してべき乗 (pow) を適用し、そのまま上書きする ("nvidia"版)
 *
 * @param[in,out] dst         GPU メモリ上のバッファ
 * @param[in]     stride_dst  dst のストライド
 * @param[in]     exponent    べき乗の指数 (double で指定、data_type に応じて変換)
 * @param[in]     n           配列の要素数
 * @param[in]     data_type   f32 または f64
 * @return ZenuStatus         成功またはエラーコード
 */
ZenuStatus zenu_compute_pow_mat_assign_nvidia(
    void*       dst,
    int         stride_dst,
    const void* exponent,
    size_t      n,
    ZenuDataType data_type
);

#ifdef __cplusplus
}
#endif


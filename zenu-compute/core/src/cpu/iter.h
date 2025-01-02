#pragma once

#include <type_traits> // for std::is_same_v
#include <cstddef>
#include "simd_iter_f32.h"
#include "simd_iter_f64.h"
// ↑ ここには質問文に示してある f32版 / f64版 の本体実装が入っている想定

//======================================================================
// (A) 1バッファ ラッパ
//======================================================================
/**
 * @brief 1バッファのラッパ
 * 
 * T が float なら f32系、double なら f64系の実装を呼び出す
 *
 * @tparam T             float or double
 * @tparam OpVecOne      ベクトル演算 (例えば __m256 operator()(__m256) など)
 * @tparam OpScalarOne   スカラー演算 (例えば float operator()(float) など)
 *
 * @param buf       [in/out] データ配列 (型T)
 * @param n         [in]     要素数
 * @param stride    [in]     ストライド(1 or >1)
 * @param opVec     SIMD用演算ラムダ
 * @param opScalar  スカラー用演算ラムダ
 * @return ZenuStatus
 */
template <typename T, class OpVecOne, class OpScalarOne>
ZenuStatus iter_1buf(
    T* buf,
    std::size_t n,
    std::size_t stride,
    OpVecOne    opVec,
    OpScalarOne opScalar)
{
    // エラーチェック
    if (!buf || n == 0) {
        return ZenuStatus::InvalidArgument;
    }
    // T が float か double かで分岐
    if constexpr (std::is_same_v<T, float>)
    {
        // stride==1 と stride!=1 の場合を切り替え
        if (stride == 1) {
            iter_1buf_f32_str1_vec_omp(buf, n, opVec, opScalar);
        } else {
            iter_1buf_f32_strN_vec_omp(buf, n, stride, opVec, opScalar);
        }
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        if (stride == 1) {
            iter_1buf_f64_str1_vec_omp(buf, n, opVec, opScalar);
        } else {
            iter_1buf_f64_strN_vec_omp(buf, n, stride, opVec, opScalar);
        }
    }
    else {
        // float, double 以外はサポート外
        return ZenuStatus::InvalidArgument;
    }

    return ZenuStatus::Success;
}

//======================================================================
// (B) 2バッファ ラッパ
//======================================================================
/**
 * @brief 2バッファのラッパ
 *
 * outB[i*strideB] = f(inA[i*strideA], outB[i*strideB]) のような処理を、
 * T が float なら f32版を、T が double なら f64版を呼び出す
 *
 * @tparam T         float or double
 * @tparam OpVec     ベクトル演算 (例: __m256 operator()(__m256, __m256))
 * @tparam OpScalar  スカラー演算 (例: float operator()(float, float))
 *
 * @param inA     [in]  入力A配列 (型T)
 * @param strideA [in]  inAのストライド
 * @param outB    [out] 出力B配列 (型T)
 * @param strideB [in]  outBのストライド
 * @param n       [in]  要素数
 * @param opVec   SIMD用演算ラムダ
 * @param opScalar スカラー用演算ラムダ
 * @return ZenuStatus
 */
template <typename T, class OpVec, class OpScalar>
ZenuStatus iter_2buf(
    const T* inA,  std::size_t strideA,
    T*       outB, std::size_t strideB,
    std::size_t n,
    OpVec     opVec,
    OpScalar  opScalar)
{
    // エラーチェック
    if (!inA || !outB || n == 0) {
        return ZenuStatus::InvalidArgument;
    }

    if constexpr (std::is_same_v<T, float>)
    {
        if (strideA == 1 && strideB == 1) {
            iter_2buf_f32_str1_vec_omp(inA, outB, n, opVec, opScalar);
        } else {
            iter_2buf_f32_strN_vec_omp(inA, strideA, outB, strideB, n, opVec, opScalar);
        }
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        if (strideA == 1 && strideB == 1) {
            iter_2buf_f64_str1_vec_omp(inA, outB, n, opVec, opScalar);
        } else {
            iter_2buf_f64_strN_vec_omp(inA, strideA, outB, strideB, n, opVec, opScalar);
        }
    }
    else {
        return ZenuStatus::InvalidArgument;
    }

    return ZenuStatus::Success;
}

//======================================================================
// (C) 3バッファ ラッパ
//======================================================================
/**
 * @brief 3バッファのラッパ
 *
 * outC[i*strideC] = f(inA[i*strideA], inB[i*strideB]) のような処理を、
 * T が float なら f32版を、T が double なら f64版を呼び出す
 *
 * @tparam T        float or double
 * @tparam OpVec    ベクトル演算 (例: __m256 operator()(__m256, __m256))
 * @tparam OpScalar スカラー演算 (例: float operator()(float, float))
 *
 * @param inA     [in]  入力A配列
 * @param strideA [in]  inAのストライド
 * @param inB     [in]  入力B配列
 * @param strideB [in]  inBのストライド
 * @param outC    [out] 出力C配列
 * @param strideC [in]  outCのストライド
 * @param n       [in]  要素数
 * @param opVec   SIMD用演算ラムダ
 * @param opScalar スカラー用演算ラムダ
 * @return ZenuStatus
 */
template <typename T, class OpVec, class OpVecStrN, class OpScalar>
ZenuStatus iter_3buf(
    const T* inA,  std::size_t strideA,
    const T* inB,  std::size_t strideB,
    T*       outC, std::size_t strideC,
    std::size_t n,
    OpVec     opVec,
    OpVecStrN opVecStrN,
    OpScalar  opScalar)
{
    // エラーチェック
    if (!inA || !inB || !outC || n == 0) {
        return ZenuStatus::InvalidArgument;
    }

    if constexpr (std::is_same_v<T, float>)
    {
        if (strideA == 1 && strideB == 1 && strideC == 1) {
            iter_3buf_f32_str1_vec_omp(inA, inB, outC, n, opVec, opScalar);
        } else {
            iter_3buf_f32_strN_vec_omp(inA, strideA, inB, strideB, outC, strideC, n, opVecStrN, opScalar);
        }
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        if (strideA == 1 && strideB == 1 && strideC == 1) {
            iter_3buf_f64_str1_vec_omp(inA, inB, outC, n, opVec, opScalar);
        } else {
            iter_3buf_f64_strN_vec_omp(inA, strideA, inB, strideB, outC, strideC, n, opVecStrN, opScalar);
        }
    }
    else {
        return ZenuStatus::InvalidArgument;
    }

    return ZenuStatus::Success;
}


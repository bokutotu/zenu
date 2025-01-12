#pragma once

#include "zenu_compute_type.h"  // ZenuDataType, ZenuStatus, ZenuTranspose が定義されている想定

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file zenu_blas.h
 * @brief CPU / NVIDIA GPU 用の GEMM (行列乗算) 関数を提供する。
 *
 * C = alpha * op(A) * op(B) + beta * C
 *   - op(A) = A   または   A^T (転置)
 *   - op(B) = B   または   B^T
 *
 * パラメータ:
 *  - transA, transB: A, B を転置するかどうか (Transpose / NoTranspose)
 *  - M, N, K: 行列の次元  
 *       - op(A) が M x K  
 *       - op(B) が K x N  
 *       - C は M x N
 *  - alpha, beta: スカラ係数 (double で指定し、f32 の場合は float にキャスト)
 *  - A, B, C: 行列データ (f32 / f64)
 *  - lda, ldb, ldc: 各行列の leading dimension (行のオフセット)
 *  - data_type: f32 または f64
 */

/**
 * @brief CPU 上で GEMM (C = alpha * op(A) * op(B) + beta * C) を実行する。
 *
 * @param[in] transA     A を転置するかどうか (Transpose / NoTranspose)
 * @param[in] transB     B を転置するかどうか (Transpose / NoTranspose)
 * @param[in] M          行列 C の行数
 * @param[in] N          行列 C の列数
 * @param[in] K          op(A) の列数 (= op(B) の行数)
 * @param[in] alpha      乗算するスカラ (double で指定; 実際に f32 の場合は float にキャスト)
 * @param[in] A          入力行列 A (CPU メモリ上)
 * @param[in] lda        A の leading dimension (op(A) に関わらず「元の行列 A」の行数で確保)
 * @param[in] B          入力行列 B (CPU メモリ上)
 * @param[in] ldb        B の leading dimension
 * @param[in] beta       C に乗算するスカラ (double で指定; 同上)
 * @param[in,out] C      出力行列 C (CPU メモリ上) - 結果は上書きされる
 * @param[in] ldc        C の leading dimension
 * @param[in] data_type  f32 または f64
 *
 * @return ZenuStatus    成功(Success) またはエラーコード
 *
 * 実装例:  
 *  - f32 の場合は、単精度で計算 (sgemm)  
 *  - f64 の場合は、倍精度で計算 (dgemm)
 *
 * @note 実装では、Eigen や OpenBLAS, 自前コードなど適宜使用可能。  
 */
ZenuStatus zenu_blas_gemm_cpu(
    ZenuTranspose transA,
    ZenuTranspose transB,
    int M,
    int N,
    int K,
    double alpha,      
    const void* A,
    int lda,
    const void* B,
    int ldb,
    double beta,
    void* C,
    int ldc,
    ZenuDataType data_type
);

/**
 * @brief NVIDIA GPU 上で GEMM (C = alpha * op(A) * op(B) + beta * C) を実行する。
 *
 * @param[in] transA     A を転置するかどうか (Transpose / NoTranspose)
 * @param[in] transB     B を転置するかどうか (Transpose / NoTranspose)
 * @param[in] M          行列 C の行数
 * @param[in] N          行列 C の列数
 * @param[in] K          op(A) の列数 (= op(B) の行数)
 * @param[in] alpha      乗算するスカラ (double で指定; 実際に f32 の場合は float にキャスト)
 * @param[in] A          入力行列 A (GPU メモリ上)
 * @param[in] lda        A の leading dimension (op(A) に関わらず「元の行列 A」の行数で確保)
 * @param[in] B          入力行列 B (GPU メモリ上)
 * @param[in] ldb        B の leading dimension
 * @param[in] beta       C に乗算するスカラ (double で指定; 同上)
 * @param[in,out] C      出力行列 C (GPU メモリ上) - 結果は上書きされる
 * @param[in] ldc        C の leading dimension
 * @param[in] data_type  f32 または f64
 *
 * @return ZenuStatus    成功(Success) またはエラーコード
 *
 * 実装例:  
 *  - f32 の場合は、cuBLAS の cublasSgemm を呼び出し  
 *  - f64 の場合は、cuBLAS の cublasDgemm を呼び出し  
 *    などの実装を想定
 *
 * @note ユーザ側で CUDA や cuBLAS の初期化 (cublasCreate など) が必要な場合もある。  
 */
ZenuStatus zenu_blas_gemm_nvidia(
    ZenuTranspose transA,
    ZenuTranspose transB,
    int M,
    int N,
    int K,
    double alpha,
    const void* A,
    int lda,
    const void* B,
    int ldb,
    double beta,
    void* C,
    int ldc,
    ZenuDataType data_type
);

#ifdef __cplusplus
}
#endif


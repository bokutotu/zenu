#include <cblas.h>
#include "zenu_compute_blas.h"
#include "zenu_compute_type.h"

static inline CBLAS_TRANSPOSE toCblasTranspose(ZenuTranspose zt)
{
    return (zt == NoTranspose) ? CblasNoTrans : CblasTrans;
}

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
)
{
    // 引数チェック (簡易)
    if (!A || !B || !C) {
        return InvalidArgument;
    }
    if (M < 0 || N < 0 || K < 0 || lda < 1 || ldb < 1 || ldc < 1) {
        return InvalidArgument;
    }

    // OpenBLAS の cblas_*gemm は
    // cblas_*gemm( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    //              const int M, const int N, const int K,
    //              const (float/double) alpha,
    //              const (float/double)* A, const int lda,
    //              const (float/double)* B, const int ldb,
    //              const (float/double) beta,
    //              (float/double)* C, const int ldc );
    //
    // ここで layout = CblasRowMajor を使うと、row-major 前提で計算できる。

    CBLAS_LAYOUT layout = CblasRowMajor;
    CBLAS_TRANSPOSE cblasTransA = toCblasTranspose(transA);
    CBLAS_TRANSPOSE cblasTransB = toCblasTranspose(transB);

    if (data_type == f32) {
        // 単精度
        float alpha_f = static_cast<float>(alpha);
        float beta_f  = static_cast<float>(beta);

        cblas_sgemm(
            layout,
            cblasTransA,
            cblasTransB,
            M,
            N,
            K,
            alpha_f,
            static_cast<const float*>(A), lda,
            static_cast<const float*>(B), ldb,
            beta_f,
            static_cast<float*>(C), ldc
        );
    } else {
        // 倍精度
        double alpha_d = alpha;
        double beta_d  = beta;

        cblas_dgemm(
            layout,
            cblasTransA,
            cblasTransB,
            M,
            N,
            K,
            alpha_d,
            static_cast<const double*>(A), lda,
            static_cast<const double*>(B), ldb,
            beta_d,
            static_cast<double*>(C), ldc
        );
    }

    return Success;
}

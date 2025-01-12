#include "zenu_compute_blas.h"
#include "handle.h"

#include <cublas_v2.h>

/**
 * row-major (Cストライド) 配列を cuBLAS に渡す場合は、実質的に転置が反転するため、
 *   - row-major で NoTranspose → col-major で Transpose
 *   - row-major で Transpose   → col-major で NoTranspose
 *
 * というマッピングを行う。
 */
static inline cublasOperation_t toCublasOpRM(ZenuTranspose zt)
{
    if (zt == NoTranspose) {
        return CUBLAS_OP_T;
    } else {
        return CUBLAS_OP_N;
    }
}

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
)
{
    if (!A || !B || !C) {
        return InvalidArgument;
    }
    if (M < 0 || N < 0 || K < 0 || lda < 1 || ldb < 1 || ldc < 1) {
        return InvalidArgument;
    }

    NvidiaHandles& handles = get_global_nvidia_handles();
    cublasHandle_t handle = *handles.cublasHandle;

    // row-major と column-major の違いにより、転置は反転させる。
    cublasOperation_t cuTransA = toCublasOpRM(transA);
    cublasOperation_t cuTransB = toCublasOpRM(transB);

    // cuBLAS では引数の順番が (transA, transB, M, N, K, ...) となるが、
    // row-major を column-major に読み替えるには M <-> N を入れ替えて呼ぶ必要がある。
    // つまり:
    //   row-major GEMM: C(M,N) = A(M,K)*B(K,N)
    //   col-major GEMM: C'(N,M) = B'(N,K)*A'(K,M)  (転置したような扱い)
    // となるため、cublasSgemm の呼び出しは下記のように "N, M, K" や "cuTransB, cuTransA" を用いる。
    //
    // 参考: NVIDIA ドキュメント "How to use cuBLAS in row-major" 等

    int cublasM = N;  // row-major の N
    int cublasN = M;  // row-major の M
    int cublasK = K;  // K は同じ

    // alpha, beta はそれぞれ float / double
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
    if (data_type == f32) {
        float alpha_f = static_cast<float>(alpha);
        float beta_f  = static_cast<float>(beta);
        stat = cublasSgemm(
            handle,
            cuTransB,  // B, A の順で転置フラグを指定
            cuTransA,
            cublasM,   // N
            cublasN,   // M
            cublasK,   // K
            &alpha_f,
            static_cast<const float*>(B), ldb,
            static_cast<const float*>(A), lda,
            &beta_f,
            static_cast<float*>(C), ldc
        );
    } else {
        double alpha_d = alpha;
        double beta_d  = beta;
        stat = cublasDgemm(
            handle,
            cuTransB,
            cuTransA,
            cublasM,
            cublasN,
            cublasK,
            &alpha_d,
            static_cast<const double*>(B), ldb,
            static_cast<const double*>(A), lda,
            &beta_d,
            static_cast<double*>(C), ldc
        );
    }

    if (stat != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy_v2(handle);
        return DeviceError;
    }

    return Success;
}


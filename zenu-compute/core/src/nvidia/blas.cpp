#include "zenu_compute_blas.h"
#include "handle.h"
#include <cublas_v2.h>

/**
 * @brief row-major (Cストライド) の A, B, C を、
 *        col-major 前提の cuBLAS で正しく計算するために、転置フラグを反転する。
 */
static inline cublasOperation_t toCublasOpRM(ZenuTranspose zt)
{
    if (zt == NoTranspose) {
        return CUBLAS_OP_N;
    } else {
        return CUBLAS_OP_T;
    }
}

/**
 * @brief row-major GEMM (C = alpha * op(A)*op(B) + beta*C) を
 *        col-major の cuBLAS で呼び出すときの引数に変換して実行する。
 *
 * row-major 上での:
 *   - A: (M x K),  lda = K
 *   - B: (K x N),  ldb = N
 *   - C: (M x N),  ldc = N
 *
 * を想定。ただし、転置フラグがある場合はさらにインデックス参照が複雑になるので
 * ここでは cuBLAS呼び出し時に M<->N を入れ替える・op(A/B) を反転する等で吸収する。
 *
 * 【注意】row-majorで (K > N) や (K > M) の場合、ldb や lda が cuBLASから見て不正にならないか、
 *         一度チェックしておくほうが安全。
 */
ZenuStatus zenu_compute_gemm_nvidia(
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

    cublasHandle_t handle = NvidiaHandles::getCublasHandle();

    cublasOperation_t cuTransA = toCublasOpRM(transA);
    cublasOperation_t cuTransB = toCublasOpRM(transB);

    int mCublas = N;
    int nCublas = M;
    int kCublas = K;

    cudaDataType AType      = CUDA_R_32F;
    cudaDataType BType      = CUDA_R_32F;
    cudaDataType CType      = CUDA_R_32F;
    cudaDataType computeType= CUDA_R_32F;

    const void* alphaPtr = nullptr;
    const void* betaPtr  = nullptr;

    float  alpha_f, beta_f;
    double alpha_d, beta_d;

    if (data_type == f32) {
        AType = BType = CType = CUDA_R_32F;
        computeType = CUDA_R_32F;

        alpha_f = static_cast<float>(alpha);
        beta_f  = static_cast<float>(beta);
        alphaPtr = &alpha_f;
        betaPtr  = &beta_f;
    }
    else if (data_type == f64) {
        AType = BType = CType = CUDA_R_64F;
        computeType = CUDA_R_64F;

        alpha_d = alpha;
        beta_d  = beta;
        alphaPtr = &alpha_d;
        betaPtr  = &beta_d;
    }
    else {
        return InvalidArgument;
    }

    cublasStatus_t stat = cublasGemmEx(
        handle,
        cuTransB,
        cuTransA,
        mCublas,
        nCublas,
        kCublas,
        alphaPtr,
        B, BType, ldb,
        A, AType, lda,
        betaPtr,
        C, CType, ldc,
        computeType,
        CUBLAS_GEMM_DEFAULT
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        return DeviceError;
    }

    return Success;
}


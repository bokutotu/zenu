#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "zenu_compute_blas.h"
#include "zenu_compute_type.h"
#include "array_comp.h"

/**
 * @brief シンプルな row-major 単精度 GEMM (NoTranspose) 実装
 * 
 * C = alpha * A * B + beta * C
 * - A: (M x K)
 * - B: (K x N)
 * - C: (M x N)
 */
static void naive_gemm_f32(
    int M, int N, int K,
    float alpha,
    const float* A, // [M*K]
    const float* B, // [K*N]
    float beta,
    float* C        // [M*N]
) {
    for(int m = 0; m < M; ++m) {
        for(int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k) {
                sum += A[m*K + k] * B[k*N + n];
            }
            // C[m, n] = alpha*sum + beta*C[m, n]
            C[m*N + n] = alpha * sum + beta * C[m*N + n];
        }
    }
}

/**
 * @brief シンプルな row-major 倍精度 GEMM (NoTranspose) 実装
 * 
 * C = alpha * A * B + beta * C
 * - A: (M x K)
 * - B: (K x N)
 * - C: (M x N)
 */
static void naive_gemm_f64(
    int M, int N, int K,
    double alpha,
    const double* A, // [M*K]
    const double* B, // [K*N]
    double beta,
    double* C        // [M*N]
) {
    for(int m = 0; m < M; ++m) {
        for(int n = 0; n < N; ++n) {
            double sum = 0.0;
            for(int k = 0; k < K; ++k) {
                sum += A[m*K + k] * B[k*N + n];
            }
            // C[m, n] = alpha*sum + beta*C[m, n]
            C[m*N + n] = alpha * sum + beta * C[m*N + n];
        }
    }
}


/**
 * @brief `zenu_blas_gemm_cpu()` の f32 実装をテスト (NoTranspose)
 *
 * alpha, beta の組み合わせを変えたり、行列サイズを変えたりする例
 */

TEST(ZenuBlasGemmTest, CPU_Gemm_F32_M2N3K4_Alpha1Beta0)
{
    // 行列サイズ
    const int M = 2;
    const int N = 3;
    const int K = 4;
    // alpha, beta
    double alpha = 1.0;
    double beta  = 0.0;

    // 入力行列・出力行列を確保 (row-major)
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    std::vector<float> C_ref(M * N);

    // データ初期化: A, B は連番, C, C_ref は -1 で初期化
    for(int i = 0; i < M*K; ++i) {
        A[i] = static_cast<float>(i);
    }
    for(int i = 0; i < K*N; ++i) {
        B[i] = static_cast<float>(i + 100);
    }
    for(int i = 0; i < M*N; ++i) {
        C[i] = -1.0f;
        C_ref[i] = -1.0f;
    }

    // === 期待値を naive_gemm_f32() で計算 ===
    naive_gemm_f32(M, N, K,
                   static_cast<float>(alpha),
                   A.data(), B.data(),
                   static_cast<float>(beta),
                   C_ref.data());

    ZenuStatus status = zenu_blas_gemm_cpu(
        NoTranspose,
        NoTranspose,
        M, N, K,
        alpha,
        A.data(),
        K,
        B.data(), 
        N,
        beta,
        C.data(),
        N,
        f32
    );
    ASSERT_EQ(status, Success);

    bool ok = array_compare(C.data(), C_ref.data(), M*N, 1e-5f);
    EXPECT_TRUE(ok) << "CPU_Gemm_F32_M2N3K4_Alpha1Beta0: result differs from naive ref!";
}

/**
 * @brief f32 テスト (別パターン: alpha=2.0, beta=1.0)
 */
TEST(ZenuBlasGemmTest, CPU_Gemm_F32_M2N3K4_Alpha2Beta1)
{
    const int M = 2;
    const int N = 3;
    const int K = 4;

    double alpha = 2.0;
    double beta  = 1.0;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    std::vector<float> C_ref(M * N);

    for(int i = 0; i < M*K; ++i) {
        A[i] = static_cast<float>(i);
    }
    for(int i = 0; i < K*N; ++i) {
        B[i] = static_cast<float>(100 + i);
    }
    for(int i = 0; i < M*N; ++i) {
        C[i] = static_cast<float>(-1 + i*0.5);   // ちょっと違う値
        C_ref[i] = C[i];                         // C_ref も同じ値で始める
    }

    // 期待値 (naive)
    naive_gemm_f32(M, N, K,
                   static_cast<float>(alpha),
                   A.data(), B.data(),
                   static_cast<float>(beta),
                   C_ref.data());

    // 実際の実装
    ZenuStatus status = zenu_blas_gemm_cpu(
        NoTranspose, NoTranspose,
        M, N, K,
        alpha,
        A.data(), K,
        B.data(), N,
        beta,
        C.data(), N,
        f32
    );
    ASSERT_EQ(status, Success);

    // 比較
    bool ok = array_compare(C.data(), C_ref.data(), M*N, 1e-5f);
    EXPECT_TRUE(ok) << "CPU_Gemm_F32_M2N3K4_Alpha2Beta1: result differs from naive ref!";
}


/**
 * @brief `zenu_blas_gemm_cpu()` の f64 実装をテスト (NoTranspose)
 *
 * 同様に倍精度版テスト
 */

TEST(ZenuBlasGemmTest, CPU_Gemm_F64_M4N2K3_Alpha1Beta0)
{
    // 例として M=4, N=2, K=3 にしてみる
    const int M = 4;
    const int N = 2;
    const int K = 3;

    double alpha = 1.0;
    double beta  = 0.0;

    std::vector<double> A(M * K);
    std::vector<double> B(K * N);
    std::vector<double> C(M * N);
    std::vector<double> C_ref(M * N);

    // 初期化
    for(int i = 0; i < M*K; ++i) {
        A[i] = static_cast<double>(i);
    }
    for(int i = 0; i < K*N; ++i) {
        B[i] = static_cast<double>(i + 500);
    }
    for(int i = 0; i < M*N; ++i) {
        C[i] = -10.0;
        C_ref[i] = -10.0;
    }

    // Naive
    naive_gemm_f64(M, N, K,
                   alpha,
                   A.data(), B.data(),
                   beta,
                   C_ref.data());

    // 実装
    ZenuStatus status = zenu_blas_gemm_cpu(
        NoTranspose, NoTranspose,
        M, N, K,
        alpha,
        A.data(), K,
        B.data(), N,
        beta,
        C.data(), N,
        f64
    );
    ASSERT_EQ(status, Success);

    bool ok = array_compare(C.data(), C_ref.data(), M*N, 1e-12);
    EXPECT_TRUE(ok) << "CPU_Gemm_F64_M4N2K3_Alpha1Beta0: result differs from naive ref!";
}

TEST(ZenuBlasGemmTest, CPU_Gemm_F64_M4N2K3_Alpha2Beta1)
{
    const int M = 4;
    const int N = 2;
    const int K = 3;

    double alpha = 2.0;
    double beta  = 1.0;

    std::vector<double> A(M * K);
    std::vector<double> B(K * N);
    std::vector<double> C(M * N);
    std::vector<double> C_ref(M * N);

    // 初期化
    for(int i = 0; i < M*K; ++i) {
        A[i] = static_cast<double>(i * 1.5);
    }
    for(int i = 0; i < K*N; ++i) {
        B[i] = static_cast<double>(1000 + i);
    }
    for(int i = 0; i < M*N; ++i) {
        C[i] = static_cast<double>(3 + i * 0.1);
        C_ref[i] = C[i];
    }

    // naive
    naive_gemm_f64(M, N, K,
                   alpha,
                   A.data(), B.data(),
                   beta,
                   C_ref.data());

    // 実装
    ZenuStatus status = zenu_blas_gemm_cpu(
        NoTranspose, NoTranspose,
        M, N, K,
        alpha,
        A.data(), K,
        B.data(), N,
        beta,
        C.data(), N,
        f64
    );
    ASSERT_EQ(status, Success);

    bool ok = array_compare(C.data(), C_ref.data(), M*N, 1e-12);
    EXPECT_TRUE(ok) << "CPU_Gemm_F64_M4N2K3_Alpha2Beta1: result differs from naive ref!";
}


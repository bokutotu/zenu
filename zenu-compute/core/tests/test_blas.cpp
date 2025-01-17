#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "zenu_compute_blas.h"
#include "zenu_compute_memory.h"
#include "zenu_compute_type.h"
#include "array_comp.h"

template <typename T>
void naive_gemm(ZenuTranspose transA,
                ZenuTranspose transB,
                int M, int N, int K,
                T alpha,
                const T* A,
                int lda,
                const T* B,
                int ldb,
                T beta,
                T* C,
                int ldc)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            T sum = (T)0;
            for (int k_ = 0; k_ < K; ++k_) {
                // Aの要素を参照
                T valA = (transA == NoTranspose)
                         ? A[m * lda + k_]
                         : A[k_ * lda + m];
                // Bの要素を参照
                T valB = (transB == NoTranspose)
                         ? B[k_ * ldb + n]
                         : B[n * ldb + k_];
                sum += valA * valB;
            }
            C[m * ldc + n] = alpha * sum + beta * C[m * ldc + n];
        }
    }
}

template <typename T>
void run_gemm_test(ZenuDataType data_type,
                   ZenuTranspose transA,
                   ZenuTranspose transB,
                   bool use_gpu,
                   int M, int N, int K,
                   double alpha,
                   double beta,
                   double tol)
{
    std::vector<T> A(M * K);
    std::vector<T> B(K * N);
    std::vector<T> C(M * N);
    std::vector<T> C_ref(M * N);

    // 適当に初期化
    for (int i = 0; i < (int)A.size(); ++i) {
        A[i] = (T)(i + 1);
    }
    for (int i = 0; i < (int)B.size(); ++i) {
        B[i] = (T)(100 + i);
    }
    for (int i = 0; i < (int)C.size(); ++i) {
        C[i] = (T)(-2 - i);
        C_ref[i] = C[i];
    }

    // 期待値
    naive_gemm<T>(
        transA, transB,
        M, N, K,
        (T)alpha,
        A.data(), (transA == NoTranspose ? K : M),
        B.data(), (transB == NoTranspose ? N : K),
        (T)beta,
        C_ref.data(),
        N
    );

    if (!use_gpu) {
        ZenuStatus st = zenu_compute_gemm_cpu(
            transA, transB,
            M, N, K,
            alpha,
            A.data(), (transA == NoTranspose ? K : M),
            B.data(), (transB == NoTranspose ? N : K),
            beta,
            C.data(), N,
            data_type
        );
        ASSERT_EQ(st, Success);
    } else {
        void* dA = nullptr;
        void* dB = nullptr;
        void* dC = nullptr;

        ZenuStatus stA = zenu_compute_malloc_nvidia(&dA, A.size() * sizeof(T));
        ZenuStatus stB = zenu_compute_malloc_nvidia(&dB, B.size() * sizeof(T));
        ZenuStatus stC = zenu_compute_malloc_nvidia(&dC, C.size() * sizeof(T));
        ASSERT_EQ(stA, Success);
        ASSERT_EQ(stB, Success);
        ASSERT_EQ(stC, Success);

        stA = zenu_compute_cpu_to_nvidia(dA, (void*)A.data(), A.size() * sizeof(T));
        stB = zenu_compute_cpu_to_nvidia(dB, (void*)B.data(), B.size() * sizeof(T));
        stC = zenu_compute_cpu_to_nvidia(dC, (void*)C.data(), C.size() * sizeof(T));
        ASSERT_EQ(stA, Success);
        ASSERT_EQ(stB, Success);
        ASSERT_EQ(stC, Success);

        ZenuStatus stG = zenu_compute_gemm_nvidia(
            transA, transB,
            M, N, K,
            alpha,
            dA, (transA == NoTranspose ? K : M),
            dB, (transB == NoTranspose ? N : K),
            beta,
            dC, N,
            data_type
        );
        ASSERT_EQ(stG, Success);

        stC = zenu_compute_nvidia_to_cpu((void*)C.data(), dC, C.size() * sizeof(T));
        ASSERT_EQ(stC, Success);

        zenu_compute_free_nvidia(dA);
        zenu_compute_free_nvidia(dB);
        zenu_compute_free_nvidia(dC);
    }

    bool ok = array_compare(C.data(), C_ref.data(), C.size(), (T)tol);
    EXPECT_TRUE(ok) << "Mismatch with transA=" << (transA==Transpose) 
                    << ", transB=" << (transB==Transpose) 
                    << ", use_gpu=" << use_gpu;
}

TEST(ZenuBlasGemmTest, CPU_f32_NoTrans) {
    run_gemm_test<float>(f32, NoTranspose, NoTranspose,
                         false, 2, 3, 4, 1.0, 0.0, 1e-5);
}

TEST(ZenuBlasGemmTest, CPU_f32_TransA) {
    run_gemm_test<float>(f32, Transpose, NoTranspose,
                         false, 2, 3, 4, 1.0, 0.0, 1e-5);
}

TEST(ZenuBlasGemmTest, CPU_f32_TransB) {
    run_gemm_test<float>(f32, NoTranspose, Transpose,
                         false, 2, 3, 4, 1.0, 0.0, 1e-5);
}

TEST(ZenuBlasGemmTest, GPU_f32_NoTrans) {
    run_gemm_test<float>(f32, NoTranspose, NoTranspose,
                         true, 2, 3, 4, 1.0, 0.0, 1e-5);
}

TEST(ZenuBlasGemmTest, GPU_f32_TransA) {
    run_gemm_test<float>(f32, Transpose, NoTranspose,
                         true, 2, 3, 4, 1.0, 0.0, 1e-5);
}

TEST(ZenuBlasGemmTest, GPU_f32_TransB) {
    run_gemm_test<float>(f32, NoTranspose, Transpose,
                         true, 2, 3, 4, 1.0, 0.0, 1e-5);
}

TEST(ZenuBlasGemmTest, CPU_f64_NoTrans) {
    run_gemm_test<double>(f64, NoTranspose, NoTranspose,
                          false, 3, 2, 5, 1.0, 0.0, 1e-12);
}

TEST(ZenuBlasGemmTest, GPU_f64_TransAB) {
    run_gemm_test<double>(f64, Transpose, Transpose,
                          true, 3, 2, 5, 1.0, 0.0, 1e-12);
}


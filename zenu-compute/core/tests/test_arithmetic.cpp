#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include "zenu_compute.h"

//======================================================================
// 1. 単一テスト生成マクロ
//    「型 (float/double)」「ストライド (1/2)」「演算 (add/sub/mul/div)」などを
//    パラメータとして、TEST(...) ブロックを生成します。
//======================================================================
#define GEN_TEST_F32_STR1(OPNAME, OP, OPSTR)                                        \
TEST(Arithmetic, MatMatMatF32Str1Cpu_##OPSTR) {                                    \
    const int N = 100;                                                             \
    const int stride = 1;                                                          \
    const float low = 0.1f;                                                        \
    const float high = 1.0f;                                                       \
    const unsigned long long seed = 1234ULL;                                       \
                                                                                   \
    std::vector<float> data1(N);                                                   \
    std::vector<float> data2(N);                                                   \
    std::vector<float> data3(N);                                                   \
                                                                                   \
    ZenuStatus st = zenu_compute_uniform_distribution_cpu(                         \
        data1.data(),                                                              \
        N,                                                                         \
        low,                                                                       \
        high,                                                                      \
        f32,                                                                       \
        seed                                                                       \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    st = zenu_compute_uniform_distribution_cpu(                                    \
        data2.data(),                                                              \
        N,                                                                         \
        low,                                                                       \
        high,                                                                      \
        f32,                                                                       \
        seed                                                                       \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    /* 例: zenu_compute_add_mat_mat_cpu(...) を演算子ごとに切り替える */          \
    st = zenu_compute_##OPNAME##_mat_mat_cpu(                                      \
        data3.data(),                                                              \
        data1.data(),                                                              \
        data2.data(),                                                              \
        stride,                                                                    \
        stride,                                                                    \
        stride,                                                                    \
        N,                                                                         \
        f32                                                                        \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    for (int i = 0; i < N; i++) {                                                  \
        /* 演算子(OP)で期待値を算出 */                                             \
        float expected = data1[i] OP data2[i];                                     \
        EXPECT_FLOAT_EQ(data3[i], expected);                                       \
    }                                                                              \
}

#define GEN_TEST_F64_STR1(OPNAME, OP, OPSTR)                                        \
TEST(Arithmetic, MatMatMatF64Str1Cpu_##OPSTR) {                                    \
    const int N = 100;                                                             \
    const int stride = 1;                                                          \
    const double low = 0.1;                                                        \
    const double high = 1.0;                                                       \
    const unsigned long long seed = 1234ULL;                                       \
                                                                                   \
    std::vector<double> data1(N);                                                  \
    std::vector<double> data2(N);                                                  \
    std::vector<double> data3(N);                                                  \
                                                                                   \
    ZenuStatus st = zenu_compute_uniform_distribution_cpu(                         \
        data1.data(),                                                              \
        N,                                                                         \
        low,                                                                       \
        high,                                                                      \
        f64,                                                                       \
        seed                                                                       \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    st = zenu_compute_uniform_distribution_cpu(                                    \
        data2.data(),                                                              \
        N,                                                                         \
        low,                                                                       \
        high,                                                                      \
        f64,                                                                       \
        seed                                                                       \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    st = zenu_compute_##OPNAME##_mat_mat_cpu(                                      \
        data3.data(),                                                              \
        data1.data(),                                                              \
        data2.data(),                                                              \
        stride,                                                                    \
        stride,                                                                    \
        stride,                                                                    \
        N,                                                                         \
        f64                                                                        \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    for (int i = 0; i < N; i++) {                                                  \
        double expected = data1[i] OP data2[i];                                    \
        EXPECT_DOUBLE_EQ(data3[i], expected);                                      \
    }                                                                              \
}

#define GEN_TEST_F32_STRN(OPNAME, OP, OPSTR)                                        \
TEST(Arithmetic, MatMatMatF32StrNCpu_##OPSTR) {                                    \
    const int N = 100;                                                             \
    const int stride = 2;                                                          \
    const float low = 0.1f;                                                        \
    const float high = 1.0f;                                                       \
    const unsigned long long seed = 1234ULL;                                       \
                                                                                   \
    std::vector<float> data1(N * stride);                                          \
    std::vector<float> data2(N * stride);                                          \
    std::vector<float> data3(N * stride);                                          \
                                                                                   \
    ZenuStatus st = zenu_compute_uniform_distribution_cpu(                         \
        data1.data(),                                                              \
        N,                                                                         \
        low,                                                                       \
        high,                                                                      \
        f32,                                                                       \
        seed                                                                       \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    st = zenu_compute_uniform_distribution_cpu(                                    \
        data2.data(),                                                              \
        N,                                                                         \
        low,                                                                       \
        high,                                                                      \
        f32,                                                                       \
        seed                                                                       \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    st = zenu_compute_##OPNAME##_mat_mat_cpu(                                      \
        data3.data(),                                                              \
        data1.data(),                                                              \
        data2.data(),                                                              \
        stride,                                                                    \
        stride,                                                                    \
        stride,                                                                    \
        N/stride,                                                                         \
        f32                                                                        \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    for (int i = 0; i < N/stride; i += stride) {                                          \
        float expected = data1[i * stride] OP data2[i * stride];                   \
        EXPECT_FLOAT_EQ(data3[i * stride], expected);                              \
    }                                                                              \
}

#define GEN_TEST_F64_STRN(OPNAME, OP, OPSTR)                                        \
TEST(Arithmetic, MatMatMatF64StrNCpu_##OPSTR) {                                    \
    const int N = 100;                                                             \
    const int stride = 2;                                                          \
    const double low = 0.1;                                                        \
    const double high = 1.0;                                                       \
    const unsigned long long seed = 1234ULL;                                       \
                                                                                   \
    std::vector<double> data1(N * stride);                                         \
    std::vector<double> data2(N * stride);                                         \
    std::vector<double> data3(N * stride);                                         \
                                                                                   \
    ZenuStatus st = zenu_compute_uniform_distribution_cpu(                         \
        data1.data(),                                                              \
        N,                                                                         \
        low,                                                                       \
        high,                                                                      \
        f64,                                                                       \
        seed                                                                       \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    st = zenu_compute_uniform_distribution_cpu(                                    \
        data2.data(),                                                              \
        N,                                                                         \
        low,                                                                       \
        high,                                                                      \
        f64,                                                                       \
        seed                                                                       \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    st = zenu_compute_##OPNAME##_mat_mat_cpu(                                      \
        data3.data(),                                                              \
        data1.data(),                                                              \
        data2.data(),                                                              \
        stride,                                                                    \
        stride,                                                                    \
        stride,                                                                    \
        N/stride,                                                                         \
        f64                                                                        \
    );                                                                              \
    ASSERT_EQ(st, Success);                                                        \
                                                                                   \
    for (int i = 0; i < N/stride; i += stride) {                                          \
        double expected = data1[i * stride] OP data2[i * stride];                  \
        EXPECT_DOUBLE_EQ(data3[i * stride], expected);                             \
    }                                                                              \
}

//======================================================================
// 2. すべての型／ストライドパターンをまとめて生成するマクロ
//    第1引数: OPNAME (関数名: add, sub, mul, div)
//    第2引数: OP     (演算子: +, -, *, / ) 
//    第3引数: OPSTR  (テスト名に付けるラベル: Add, Sub, Mul, Div など)
//======================================================================
#define GEN_ALL_TESTS_FOR_OP(OPNAME, OP, OPSTR) \
    GEN_TEST_F32_STR1(OPNAME, OP, OPSTR)        \
    GEN_TEST_F64_STR1(OPNAME, OP, OPSTR)        \
    GEN_TEST_F32_STRN(OPNAME, OP, OPSTR)        \
    GEN_TEST_F64_STRN(OPNAME, OP, OPSTR)

//======================================================================
// 3. 実際に add, sub, mul, div のテストを生成するマクロ呼び出し
//    - 第一引数: 実際の「zenu_compute_xxx_mat_mat_cpu」の xxx 名 (add / sub / mul / div)
//    - 第二引数: C++ の演算子
//    - 第三引数: テスト名ラベル (Add / Sub / Mul / Div)
//======================================================================
GEN_ALL_TESTS_FOR_OP(add, +, Add)
GEN_ALL_TESTS_FOR_OP(sub, -, Sub)
GEN_ALL_TESTS_FOR_OP(mul, *, Mul)
GEN_ALL_TESTS_FOR_OP(div, /, Div)


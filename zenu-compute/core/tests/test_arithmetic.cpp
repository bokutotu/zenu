#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include "zenu_compute.h"

TEST(Add, MatMatMatF32Str1Cpu) {
    const int N = 100;
    const int stride = 1;
    const float scalar = 1.0f;
    const float low = 0.0f;
    const float high = 1.0f;
    const unsigned long long seed = 1234ULL;

    std::vector<float> data1(N);
    std::vector<float> data2(N);
    std::vector<float> data3(N);

    ZenuStatus st = zenu_compute_uniform_distribution_cpu(
        data1.data(),
        N,
        low,
        high,
        f32,
        seed
    );
    ASSERT_EQ(st, Success);

    st = zenu_compute_uniform_distribution_cpu(
        data2.data(),
        N,
        low,
        high,
        f32,
        seed
    );
    ASSERT_EQ(st, Success);

    st = zenu_compute_add_mat_mat_cpu(
        data3.data(),
        data1.data(),
        data2.data(),
        stride,
        stride,
        stride,
        N,
        f32
    );
    ASSERT_EQ(st, Success);

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(data3[i], data1[i] + data2[i]);
    }
}

TEST(Add, MatMatMatF64Str1Cpu) {
    const int N = 100;
    const int stride = 1;
    const double scalar = 1.0;
    const double low = 0.0;
    const double high = 1.0;
    const unsigned long long seed = 1234ULL;

    std::vector<double> data1(N);
    std::vector<double> data2(N);
    std::vector<double> data3(N);

    ZenuStatus st = zenu_compute_uniform_distribution_cpu(
        data1.data(),
        N,
        low,
        high,
        f64,
        seed
    );
    ASSERT_EQ(st, Success);

    st = zenu_compute_uniform_distribution_cpu(
        data2.data(),
        N,
        low,
        high,
        f64,
        seed
    );
    ASSERT_EQ(st, Success);

    st = zenu_compute_add_mat_mat_cpu(
        data3.data(),
        data1.data(),
        data2.data(),
        stride,
        stride,
        stride,
        N,
        f64
    );
    ASSERT_EQ(st, Success);

    for (int i = 0; i < N; i++) {
        EXPECT_DOUBLE_EQ(data3[i], data1[i] + data2[i]);
    }
}

TEST(Add, MatMatMatF32StrNCpu) {
    const int N = 100;
    const int stride = 2;
    const float scalar = 1.0f;
    const float low = 0.0f;
    const float high = 1.0f;
    const unsigned long long seed = 1234ULL;

    std::vector<float> data1(N * stride);
    std::vector<float> data2(N * stride);
    std::vector<float> data3(N * stride);

    ZenuStatus st = zenu_compute_uniform_distribution_cpu(
        data1.data(),
        N,
        low,
        high,
        f32,
        seed
    );
    ASSERT_EQ(st, Success);

    st = zenu_compute_uniform_distribution_cpu(
        data2.data(),
        N,
        low,
        high,
        f32,
        seed
    );
    ASSERT_EQ(st, Success);

    st = zenu_compute_add_mat_mat_cpu(
        data3.data(),
        data1.data(),
        data2.data(),
        stride,
        stride,
        stride,
        N,
        f32
    );
    ASSERT_EQ(st, Success);

    for (int i = 0; i < N; i+=stride) {
        EXPECT_FLOAT_EQ(data3[i * stride], data1[i * stride] + data2[i * stride]);
    }
}

TEST(Add, MatMatMatF64StrNCpu) {
    const int N = 100;
    const int stride = 2;
    const double scalar = 1.0;
    const double low = 0.0;
    const double high = 1.0;
    const unsigned long long seed = 1234ULL;

    std::vector<double> data1(N * stride);
    std::vector<double> data2(N * stride);
    std::vector<double> data3(N * stride);

    ZenuStatus st = zenu_compute_uniform_distribution_cpu(
        data1.data(),
        N,
        low,
        high,
        f64,
        seed
    );
    ASSERT_EQ(st, Success);

    st = zenu_compute_uniform_distribution_cpu(
        data2.data(),
        N,
        low,
        high,
        f64,
        seed
    );
    ASSERT_EQ(st, Success);

    st = zenu_compute_add_mat_mat_cpu(
        data3.data(),
        data1.data(),
        data2.data(),
        stride,
        stride,
        stride,
        N,
        f64
    );
    ASSERT_EQ(st, Success);

    for (int i = 0; i < N; i+=stride) {
        EXPECT_DOUBLE_EQ(data3[i * stride], data1[i * stride] + data2[i * stride]);
    }
}

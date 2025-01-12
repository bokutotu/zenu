#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

#include "zenu_compute.h"

#include <cuda_runtime.h>

// 比較用の許容誤差
constexpr float  FLOAT_EPSILON  = 1e-5f;
constexpr double DOUBLE_EPSILON = 1e-12;

/**
 * @brief CPU版 sin out-of-place: dst = sin(src) のテスト
 */
TEST(ZenuComputeSinCpuTest, OutOfPlaceFloat)
{
    // 入力データを適当に用意
    std::vector<float> src = {0.f, float(M_PI/6), float(M_PI/4), float(M_PI/3), float(M_PI/2)};
    std::vector<float> dst(src.size(), 0.f);

    // stride = 1 とし、連続メモリ上に格納
    int stride_src = 1;
    int stride_dst = 1;
    size_t n = src.size();

    // テスト呼び出し
    ZenuStatus status = zenu_compute_sin_mat_cpu(
        dst.data(),
        src.data(),
        stride_dst,
        stride_src,
        n,
        f32   // float
    );

    // 成功を確認
    ASSERT_EQ(status, Success);

    // 結果を検証
    for (size_t i = 0; i < n; ++i) {
        float expected = std::sin(src[i]);
        EXPECT_NEAR(dst[i], expected, FLOAT_EPSILON) 
            << "i=" << i << ", src=" << src[i] << ", dst=" << dst[i];
    }
}

/**
 * @brief CPU版 sin in-place: dst = sin(dst) のテスト
 */
TEST(ZenuComputeSinCpuTest, InPlaceFloat)
{
    // 入力データを適当に用意
    std::vector<float> dst = {0.f, float(M_PI/6), float(M_PI/4), float(M_PI/3), float(M_PI/2)};
    std::vector<float> ans;
    for (auto x : dst) {
        ans.push_back(std::sin(x));
    }

    int stride_dst = 1;
    size_t n = dst.size();

    ZenuStatus status = zenu_compute_sin_mat_assign_cpu(
        dst.data(),
        stride_dst,
        n,
        f32
    );
    ASSERT_EQ(status, Success);

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(dst[i], ans[i], FLOAT_EPSILON) 
            << "i=" << i << ", dst=" << dst[i] << ", ans=" << ans[i];
    }
}

/**
 * @brief CPU版 cos out-of-place: dst = cos(src) のテスト (double精度)
 */
TEST(ZenuComputeCosCpuTest, OutOfPlaceDouble)
{
    std::vector<double> src = {0.0, M_PI/6, M_PI/4, M_PI/3, M_PI/2};
    std::vector<double> dst(src.size(), 0.0);

    int stride_src = 1;
    int stride_dst = 1;
    size_t n = src.size();

    ZenuStatus status = zenu_compute_cos_mat_cpu(
        dst.data(),
        src.data(),
        stride_dst,
        stride_src,
        n,
        f64
    );
    ASSERT_EQ(status, Success);

    for (size_t i = 0; i < n; ++i) {
        double expected = std::cos(src[i]);
        EXPECT_NEAR(dst[i], expected, DOUBLE_EPSILON) 
            << "i=" << i << ", src=" << src[i];
    }
}

/**
 * @brief CPU版 cos in-place: dst = cos(dst) のテスト (double精度)
 */
TEST(ZenuComputeCosCpuTest, InPlaceDouble)
{
    std::vector<double> dst = {0.0, M_PI/6, M_PI/4, M_PI/3, M_PI/2};

    int stride_dst = 1;
    size_t n = dst.size();

    // 事前コピーしておく
    std::vector<double> original = dst;

    ZenuStatus status = zenu_compute_cos_mat_assign_cpu(
        dst.data(),
        stride_dst,
        n,
        f64
    );
    ASSERT_EQ(status, Success);

    for (size_t i = 0; i < n; ++i) {
        double expected = std::cos(original[i]);
        EXPECT_NEAR(dst[i], expected, DOUBLE_EPSILON) 
            << "i=" << i;
    }
}



/**
 * @brief GPU版 tan out-of-place: dst = tan(src) のテスト (float精度)
 */
TEST(ZenuComputeTanNvidiaTest, OutOfPlaceFloat)
{
    std::vector<float> host_src = {0.f, float(M_PI/6), float(M_PI/4), float(M_PI/3)};
    std::vector<float> host_dst(host_src.size(), 0.f);

    const size_t n = host_src.size();
    const size_t bytes = n * sizeof(float);
    int stride_src = 1;
    int stride_dst = 1;

    // GPUメモリ確保 (zenu_compute_malloc_nvidia)
    float* d_src = nullptr;
    float* d_dst = nullptr;

    ZenuStatus status = zenu_compute_malloc_nvidia((void**)&d_src, bytes);
    ASSERT_EQ(status, Success) << "Failed to allocate GPU memory for d_src";

    status = zenu_compute_malloc_nvidia((void**)&d_dst, bytes);
    ASSERT_EQ(status, Success) << "Failed to allocate GPU memory for d_dst";

    // CPU -> GPU 転送 (zenu_compute_cpu_to_nvidia)
    status = zenu_compute_cpu_to_nvidia(d_src, host_src.data(), bytes);
    ASSERT_EQ(status, Success) << "Failed to copy data from CPU to GPU (d_src)";

    // テスト呼び出し (tan out-of-place)
    status = zenu_compute_tan_mat_nvidia(
        d_dst,   // 出力 GPUバッファ
        d_src,   // 入力 GPUバッファ
        stride_dst,
        stride_src,
        n,
        f32
    );
    ASSERT_EQ(status, Success);

    // GPU -> CPU 転送 (zenu_compute_nvidia_to_cpu)
    status = zenu_compute_nvidia_to_cpu(host_dst.data(), d_dst, bytes);
    ASSERT_EQ(status, Success) << "Failed to copy data from GPU to CPU (d_dst)";

    // 結果を検証
    for (size_t i = 0; i < n; ++i) {
        float expected = std::tan(host_src[i]);
        EXPECT_NEAR(host_dst[i], expected, FLOAT_EPSILON) 
            << "i=" << i << ", src=" << host_src[i];
    }

    // GPUメモリ開放
    zenu_compute_free_nvidia(d_src);
    zenu_compute_free_nvidia(d_dst);
}

/**
 * @brief GPU版 tan in-place: dst = tan(dst) のテスト (float精度)
 */
TEST(ZenuComputeTanNvidiaTest, InPlaceFloat)
{
    std::vector<float> host_dst = {0.f, float(M_PI/6), float(M_PI/4), float(M_PI/3)};

    const size_t n = host_dst.size();
    const size_t bytes = n * sizeof(float);
    int stride_dst = 1;

    // GPUメモリ確保
    float* d_dst = nullptr;
    ZenuStatus status = zenu_compute_malloc_nvidia((void**)&d_dst, bytes);
    ASSERT_EQ(status, Success) << "Failed to allocate GPU memory for d_dst";

    // CPU -> GPU
    status = zenu_compute_cpu_to_nvidia(d_dst, host_dst.data(), bytes);
    ASSERT_EQ(status, Success) << "Failed to copy data from CPU to GPU (d_dst)";

    // 事前にコピーした元データ (検証用)
    std::vector<float> original = host_dst;

    // テスト呼び出し (tan in-place)
    status = zenu_compute_tan_mat_assign_nvidia(
        d_dst,   // dst: GPUバッファ
        stride_dst,
        n,
        f32
    );
    ASSERT_EQ(status, Success);

    // GPU -> CPU
    status = zenu_compute_nvidia_to_cpu(host_dst.data(), d_dst, bytes);
    ASSERT_EQ(status, Success) << "Failed to copy data from GPU to CPU (d_dst)";

    // 結果確認
    for (size_t i = 0; i < n; ++i) {
        float expected = std::tan(original[i]);
        EXPECT_NEAR(host_dst[i], expected, FLOAT_EPSILON) 
            << "i=" << i << ", original=" << original[i];
    }

    // GPUメモリ開放
    zenu_compute_free_nvidia(d_dst);
}

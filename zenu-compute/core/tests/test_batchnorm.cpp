// test_batchnorm.cpp
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include "zenu_compute_norm.h"
#include "zenu_compute_memory.h"
#include "zenu_compute_random.h"
#include "zenu_compute_type.h"
#include "array_comp.h"

// Naive 実装（inference）: 入力 shape は {N, C, H, W} と仮定
template <typename T>
void batchnorm_forward_inference_naive(
    const T* input,
    const T* scale,
    const T* bias,
    const T* mean,
    const T* inv_variance,
    T* output,
    const std::vector<size_t>& input_shape)
{
    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    size_t idx = n * C * H * W + c * H * W + h * W + w;
                    output[idx] = scale[c] * ((input[idx] - mean[c]) * inv_variance[c]) + bias[c];
                }
            }
        }
    }
}

// Naive 実装（forward train）
// 各チャンネル毎に mean と分散を計算し、inv_variance と出力を求める
template <typename T>
void batchnorm_forward_train_naive(
    const T* input,
    const T* scale,
    const T* bias,
    T* output,
    T* mean_out,
    T* inv_variance_out,
    const std::vector<size_t>& input_shape,
    T epsilon = static_cast<T>(1e-5))
{
    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];
    size_t M = N * H * W;
    std::vector<T> mean(C, 0), var(C, 0);

    // 平均の計算
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    size_t idx = n * C * H * W + c * H * W + h * W + w;
                    mean[c] += input[idx];
                }
            }
        }
    }
    for (size_t c = 0; c < C; ++c) {
        mean[c] /= static_cast<T>(M);
        mean_out[c] = mean[c];
    }

    // 分散の計算
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    size_t idx = n * C * H * W + c * H * W + h * W + w;
                    T diff = input[idx] - mean[c];
                    var[c] += diff * diff;
                }
            }
        }
    }
    for (size_t c = 0; c < C; ++c) {
        var[c] /= static_cast<T>(M);
        inv_variance_out[c] = static_cast<T>(1) / std::sqrt(var[c] + epsilon);
    }

    // 出力の計算
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    size_t idx = n * C * H * W + c * H * W + h * W + w;
                    output[idx] = scale[c] * ((input[idx] - mean[c]) * inv_variance_out[c]) + bias[c];
                }
            }
        }
    }
}

// Naive 実装（backward）
// バッチ正規化の逆伝播：d_bias, d_scale, d_input を各チャンネル毎に計算する
template <typename T>
void batchnorm_backward_naive(
    const T* d_output,
    const T* input,
    const T* scale,
    const T* mean,
    const T* inv_variance,
    T* d_input,
    T* d_scale,
    T* d_bias,
    const std::vector<size_t>& input_shape)
{
    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];
    size_t M = N * H * W;

    // 初期化
    std::fill(d_bias, d_bias + C, static_cast<T>(0));
    std::fill(d_scale, d_scale + C, static_cast<T>(0));
    std::fill(d_input, d_input + N * C * H * W, static_cast<T>(0));

    // d_bias と d_scale をチャンネル毎に計算
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    size_t idx = n * C * H * W + c * H * W + h * W + w;
                    d_bias[c] += d_output[idx];
                    T x_hat = (input[idx] - mean[c]) * inv_variance[c];
                    d_scale[c] += d_output[idx] * x_hat;
                }
            }
        }
    }

    // d_input の計算
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    size_t idx = n * C * H * W + c * H * W + h * W + w;
                    T x_hat = (input[idx] - mean[c]) * inv_variance[c];
                    d_input[idx] = scale[c] * inv_variance[c] / static_cast<T>(M) *
                        (static_cast<T>(M) * d_output[idx] - d_bias[c] - x_hat * d_scale[c]);
                }
            }
        }
    }
}

//---------------------------------------------------------------------
// Forward Inference テスト
TEST(ZenuBatchNormTest, ForwardInferenceTest) {
    // 入力形状: {N, C, H, W} = {4, 3, 5, 5}
    std::vector<size_t> input_shape = {4, 3, 5, 5};
    size_t total_elements = 4 * 3 * 5 * 5;
    size_t param_size = 3; // チャンネル数

    // BatchNorm ハンドル作成
    ZenuComputeBatchNorm* batchnorm = nullptr;
    zenu_compute_create_batchnorm(&batchnorm);

    // inference モード (is_train = false) で初期化
    ZenuStatus st = zenu_compute_init_batchnorm(batchnorm, input_shape.data(), input_shape.size(), f32, false);
    ASSERT_EQ(st, ZenuStatus::Success);

    // 必要なメモリの確保
    void *input_buf, *scale_buf, *bias_buf, *mean_buf, *inv_variance_buf, *output_buf, *workspace;
    zenu_compute_malloc_cpu(&input_buf, total_elements * sizeof(float));
    zenu_compute_malloc_cpu(&scale_buf, param_size * sizeof(float));
    zenu_compute_malloc_cpu(&bias_buf, param_size * sizeof(float));
    zenu_compute_malloc_cpu(&mean_buf, param_size * sizeof(float));
    zenu_compute_malloc_cpu(&inv_variance_buf, param_size * sizeof(float));
    zenu_compute_malloc_cpu(&output_buf, total_elements * sizeof(float));
    size_t ws_bytes = zenu_compute_batchnorm_forward_get_workspace_bytes(batchnorm);
    zenu_compute_malloc_cpu(&workspace, ws_bytes);

    // 入力、scale、bias、mean、inv_variance を乱数で初期化
    st = zenu_compute_normal_distribution_cpu(input_buf, total_elements, 0.0f, 1.0f, f32, 1234);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(scale_buf, param_size, 0.0f, 1.0f, f32, 2345);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(bias_buf, param_size, 0.0f, 1.0f, f32, 3456);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(mean_buf, param_size, 0.0f, 1.0f, f32, 4567);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(inv_variance_buf, param_size, 0.0f, 1.0f, f32, 5678);
    ASSERT_EQ(st, ZenuStatus::Success);

    // ライブラリの inference forward を実行
    st = zenu_compute_forward_batchnorm_inference(
        batchnorm,
        input_buf,
        scale_buf,
        bias_buf,
        mean_buf,
        inv_variance_buf,
        output_buf,
        workspace
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // Naive 実装による出力計算
    std::vector<float> ref_output(total_elements, 0);
    batchnorm_forward_inference_naive<float>(
        static_cast<const float*>(input_buf),
        static_cast<const float*>(scale_buf),
        static_cast<const float*>(bias_buf),
        static_cast<const float*>(mean_buf),
        static_cast<const float*>(inv_variance_buf),
        ref_output.data(),
        input_shape
    );

    const float* lib_output = static_cast<const float*>(output_buf);
    ASSERT_TRUE(array_compare(lib_output, ref_output.data(), total_elements, 1e-5f));

    // 後始末
    zenu_compute_free_cpu(input_buf);
    zenu_compute_free_cpu(scale_buf);
    zenu_compute_free_cpu(bias_buf);
    zenu_compute_free_cpu(mean_buf);
    zenu_compute_free_cpu(inv_variance_buf);
    zenu_compute_free_cpu(output_buf);
    zenu_compute_free_cpu(workspace);
    zenu_compute_destroy_batchnorm(batchnorm);
}

//---------------------------------------------------------------------
// Forward Train テスト
TEST(ZenuBatchNormTest, ForwardTrainTest) {
    // 入力形状: {4, 3, 5, 5}
    std::vector<size_t> input_shape = {4, 3, 5, 5};
    size_t total_elements = 4 * 3 * 5 * 5;
    size_t param_size = 3;

    // BatchNorm ハンドル作成
    ZenuComputeBatchNorm* batchnorm = nullptr;
    zenu_compute_create_batchnorm(&batchnorm);

    // training モード (is_train = true) で初期化
    ZenuStatus st = zenu_compute_init_batchnorm(batchnorm, input_shape.data(), input_shape.size(), f32, true);
    ASSERT_EQ(st, ZenuStatus::Success);

    // 必要なメモリの確保
    void *input_buf, *scale_buf, *bias_buf, *output_buf, *mean_buf, *inv_variance_buf, *workspace;
    zenu_compute_malloc_cpu(&input_buf, total_elements * sizeof(float));
    zenu_compute_malloc_cpu(&scale_buf, param_size * sizeof(float));
    zenu_compute_malloc_cpu(&bias_buf, param_size * sizeof(float));
    zenu_compute_malloc_cpu(&output_buf, total_elements * sizeof(float));
    zenu_compute_malloc_cpu(&mean_buf, param_size * sizeof(float));
    zenu_compute_malloc_cpu(&inv_variance_buf, param_size * sizeof(float));
    size_t ws_bytes = zenu_compute_batchnorm_forward_get_workspace_bytes(batchnorm);
    zenu_compute_malloc_cpu(&workspace, ws_bytes);

    // 入力、scale、bias を初期化
    st = zenu_compute_normal_distribution_cpu(input_buf, total_elements, 0.0f, 1.0f, f32, 6789);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(scale_buf, param_size, 0.0f, 1.0f, f32, 7890);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(bias_buf, param_size, 0.0f, 1.0f, f32, 8901);
    ASSERT_EQ(st, ZenuStatus::Success);

    // ライブラリの forward train を実行
    st = zenu_compute_forward_batchnorm_train(
        batchnorm,
        input_buf,
        scale_buf,
        bias_buf,
        output_buf,
        mean_buf,
        inv_variance_buf,
        workspace
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // Naive 実装による出力、mean, inv_variance の計算
    std::vector<float> ref_output(total_elements, 0);
    std::vector<float> ref_mean(param_size, 0);
    std::vector<float> ref_inv_variance(param_size, 0);
    batchnorm_forward_train_naive<float>(
        static_cast<const float*>(input_buf),
        static_cast<const float*>(scale_buf),
        static_cast<const float*>(bias_buf),
        ref_output.data(),
        ref_mean.data(),
        ref_inv_variance.data(),
        input_shape
    );

    const float* lib_output = static_cast<const float*>(output_buf);
    const float* lib_mean = static_cast<const float*>(mean_buf);
    const float* lib_inv_variance = static_cast<const float*>(inv_variance_buf);

    ASSERT_TRUE(array_compare(lib_output, ref_output.data(), total_elements, 1e-4f));
    ASSERT_TRUE(array_compare(lib_mean, ref_mean.data(), param_size, 1e-4f));
    ASSERT_TRUE(array_compare(lib_inv_variance, ref_inv_variance.data(), param_size, 1e-4f));

    // 後始末
    zenu_compute_free_cpu(input_buf);
    zenu_compute_free_cpu(scale_buf);
    zenu_compute_free_cpu(bias_buf);
    zenu_compute_free_cpu(output_buf);
    zenu_compute_free_cpu(mean_buf);
    zenu_compute_free_cpu(inv_variance_buf);
    zenu_compute_free_cpu(workspace);
    zenu_compute_destroy_batchnorm(batchnorm);
}

//---------------------------------------------------------------------
// Backward テスト
TEST(ZenuBatchNormTest, BackwardTest) {
    // 入力形状: {4, 3, 5, 5}
    std::vector<size_t> input_shape = {4, 3, 5, 5};
    size_t total_elements = 4 * 3 * 5 * 5;
    size_t param_size = 3;

    // BatchNorm ハンドル作成
    ZenuComputeBatchNorm* batchnorm = nullptr;
    zenu_compute_create_batchnorm(&batchnorm);

    // training モードで初期化
    ZenuStatus st = zenu_compute_init_batchnorm(batchnorm, input_shape.data(), input_shape.size(), f32, true);
    ASSERT_EQ(st, ZenuStatus::Success);

    // 順伝播・逆伝播に必要なメモリを確保
    void *input_buf, *scale_buf, *bias_buf, *d_output_buf;
    void *d_input_buf, *d_scale_buf, *d_bias_buf, *mean_buf, *inv_variance_buf, *workspace;
    zenu_compute_malloc_cpu(&input_buf, total_elements * sizeof(float));
    zenu_compute_malloc_cpu(&scale_buf, param_size * sizeof(float));
    zenu_compute_malloc_cpu(&bias_buf, param_size * sizeof(float));
    zenu_compute_malloc_cpu(&d_output_buf, total_elements * sizeof(float));
    zenu_compute_malloc_cpu(&d_input_buf, total_elements * sizeof(float));
    zenu_compute_malloc_cpu(&d_scale_buf, param_size * sizeof(float));
    zenu_compute_malloc_cpu(&d_bias_buf, param_size * sizeof(float));
    // 順伝播で計算される mean, inv_variance
    zenu_compute_malloc_cpu(&mean_buf, param_size * sizeof(float));
    zenu_compute_malloc_cpu(&inv_variance_buf, param_size * sizeof(float));
    size_t bw_ws_bytes = zenu_compute_batchnorm_backward_get_workspace_bytes(batchnorm);
    zenu_compute_malloc_cpu(&workspace, bw_ws_bytes);

    // 入力、scale、bias、d_output を乱数で初期化
    st = zenu_compute_normal_distribution_cpu(input_buf, total_elements, 0.0f, 1.0f, f32, 9012);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(scale_buf, param_size, 0.0f, 1.0f, f32, 1234);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(bias_buf, param_size, 0.0f, 1.0f, f32, 2345);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(d_output_buf, total_elements, 0.0f, 1.0f, f32, 3456);
    ASSERT_EQ(st, ZenuStatus::Success);

    // 順伝播を実行して、mean と inv_variance を計算（出力は不要）
    void* dummy_output;
    zenu_compute_malloc_cpu(&dummy_output, total_elements * sizeof(float));
    st = zenu_compute_forward_batchnorm_train(
        batchnorm,
        input_buf,
        scale_buf,
        bias_buf,
        dummy_output,
        mean_buf,
        inv_variance_buf,
        workspace
    );
    ASSERT_EQ(st, ZenuStatus::Success);
    zenu_compute_free_cpu(dummy_output);

    // ライブラリの backward を実行
    st = zenu_compute_backward_batchnorm(
        batchnorm,
        d_output_buf,
        input_buf,
        scale_buf,
        mean_buf,
        inv_variance_buf,
        d_input_buf,
        d_scale_buf,
        d_bias_buf,
        workspace
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // Naive 実装による各勾配の計算
    std::vector<float> ref_d_input(total_elements, 0);
    std::vector<float> ref_d_scale(param_size, 0);
    std::vector<float> ref_d_bias(param_size, 0);
    batchnorm_backward_naive<float>(
        static_cast<const float*>(d_output_buf),
        static_cast<const float*>(input_buf),
        static_cast<const float*>(scale_buf),
        static_cast<const float*>(mean_buf),
        static_cast<const float*>(inv_variance_buf),
        ref_d_input.data(),
        ref_d_scale.data(),
        ref_d_bias.data(),
        input_shape
    );

    const float* lib_d_input = static_cast<const float*>(d_input_buf);
    const float* lib_d_scale = static_cast<const float*>(d_scale_buf);
    const float* lib_d_bias = static_cast<const float*>(d_bias_buf);

    ASSERT_TRUE(array_compare(lib_d_input, ref_d_input.data(), total_elements, 1e-4f));
    ASSERT_TRUE(array_compare(lib_d_scale, ref_d_scale.data(), param_size, 1e-4f));
    ASSERT_TRUE(array_compare(lib_d_bias, ref_d_bias.data(), param_size, 1e-4f));

    // 後始末
    zenu_compute_free_cpu(input_buf);
    zenu_compute_free_cpu(scale_buf);
    zenu_compute_free_cpu(bias_buf);
    zenu_compute_free_cpu(d_output_buf);
    zenu_compute_free_cpu(d_input_buf);
    zenu_compute_free_cpu(d_scale_buf);
    zenu_compute_free_cpu(d_bias_buf);
    zenu_compute_free_cpu(mean_buf);
    zenu_compute_free_cpu(inv_variance_buf);
    zenu_compute_free_cpu(workspace);
    zenu_compute_destroy_batchnorm(batchnorm);
}


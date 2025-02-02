#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include "zenu_compute_conv.h"
#include "zenu_compute_memory.h"
#include "zenu_compute_random.h"
#include "zenu_compute_type.h"
#include "array_comp.h"

/**
 * @brief Naive な 1D 畳み込み実装（入力は NCW 順）
 * @tparam T データ型 (float/double)
 * @param input 入力テンソル (NCW: {N, C, L})
 * @param kernel カーネル (フィルタ) テンソル (KCS: {K, C, S})
 * @param output 出力テンソル (NCW: {N, K, L_out})
 * @param input_shape 入力形状 {N, C, L}
 * @param kernel_shape カーネル形状 {K, C, S}
 * @param output_shape 出力形状 {N, K, L_out}
 * @param stride ストライド {stride}
 * @param padding パディング {pad}
 * @param dilation ダイレーション {dilation}
 */
template <typename T>
void conv1d_naive(
    const T* input,
    const T* kernel,
    T* output,
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& kernel_shape,
    const std::vector<size_t>& output_shape,
    const std::vector<size_t>& stride,
    const std::vector<size_t>& padding,
    const std::vector<size_t>& dilation)
{
    const size_t N = input_shape[0];
    const size_t C = input_shape[1];
    const size_t L = input_shape[2];

    const size_t K = kernel_shape[0];
    const size_t S = kernel_shape[2];

    const size_t L_out = output_shape[2];

    // ループ: バッチ, フィルタ, 出力位置
    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t l = 0; l < L_out; ++l) {
                T sum = 0;
                // 各入力チャネルおよびカーネル幅に対して畳み込み計算
                for (size_t c = 0; c < C; ++c) {
                    for (size_t s = 0; s < S; ++s) {
                        // 入力位置の算出（ストライド、パディング、ダイレーションを考慮）
                        int x = static_cast<int>(l * stride[0]) - static_cast<int>(padding[0]) + static_cast<int>(s * dilation[0]);
                        if (x >= 0 && x < static_cast<int>(L)) {
                            const size_t input_idx = n * C * L + c * L + x;
                            const size_t kernel_idx = k * C * S + c * S + s;
                            sum += input[input_idx] * kernel[kernel_idx];
                        }
                    }
                }
                const size_t output_idx = n * K * L_out + k * L_out + l;
                output[output_idx] = sum;
            }
        }
    }
}

/**
 * @brief CPU による 1D 畳み込み forward テスト (float32)
 *
 * 入力は (N, C, L) = (2, 3, 28)、カーネルは (K, C, S) = (2, 3, 5)、
 * 出力は (N, K, L_out) = (2, 2, 28) とし、パディングは 2、ストライド・ダイレーションは 1 としています。<br>
 * ライブラリ実装による結果と naive 実装による結果が一致するか検証します。
 */
TEST(ZenuConvCpuTest, FloatForward1DConvTest) {
    // 畳み込みパラメータの設定
    const std::vector<size_t> input_shape  = {2, 3, 28};  // N, C, L
    const std::vector<size_t> kernel_shape = {2, 3, 5};    // K, C, S
    const std::vector<size_t> output_shape = {2, 2, 28};   // N, K, L_out
    const std::vector<size_t> stride       = {1};          // ストライド
    const std::vector<size_t> padding      = {2};          // パディング
    const std::vector<size_t> dilation     = {1};          // ダイレーション

    // CPU 畳み込みハンドルの作成
    ZenuComputeConvCpu* conv_cpu = nullptr;
    ZenuStatus st = zenu_compute_create_conv_cpu(&conv_cpu);
    ASSERT_EQ(st, ZenuStatus::Success);

    // 畳み込みディスクリプタの設定（1D 畳み込みの場合、num_dim に 1 を指定）
    st = zenu_compute_set_conv_cpu_descriptor_cpu(
        conv_cpu,
        const_cast<size_t*>(input_shape.data()),
        const_cast<size_t*>(output_shape.data()),
        const_cast<size_t*>(kernel_shape.data()),
        const_cast<size_t*>(stride.data()),   // ※ test_conv2d.cpp の順序に合わせています
        const_cast<size_t*>(padding.data()),
        const_cast<size_t*>(dilation.data()),
        ZenuDataType::f32,
        1
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // メモリ領域の確保
    const size_t input_size  = input_shape[0] * input_shape[1] * input_shape[2];
    const size_t kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
    const size_t output_size = output_shape[0] * output_shape[1] * output_shape[2];

    void* input_buf  = nullptr;
    void* kernel_buf = nullptr;
    void* output_buf = nullptr;
    void* workspace  = nullptr;

    zenu_compute_malloc_cpu(&input_buf,  input_size  * sizeof(float));
    zenu_compute_malloc_cpu(&kernel_buf, kernel_size * sizeof(float));
    zenu_compute_malloc_cpu(&output_buf, output_size * sizeof(float));

    // ワークスペースのサイズ取得と確保
    const size_t workspace_bytes = zenu_compute_conv_get_forward_workspace_bytes_cpu(conv_cpu);
    zenu_compute_malloc_cpu(&workspace, workspace_bytes);

    // 入力およびカーネルの初期化（正規分布）
    st = zenu_compute_normal_distribution_cpu(input_buf,  input_size, 0.0f, 1.0f, ZenuDataType::f32, 1234);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(kernel_buf, kernel_size, 0.0f, 1.0f, ZenuDataType::f32, 5678);
    ASSERT_EQ(st, ZenuStatus::Success);

    // naive 実装による参照結果の計算
    std::vector<float> naive_output(output_size, 0.0f);
    conv1d_naive<float>(
        static_cast<const float*>(input_buf),
        static_cast<const float*>(kernel_buf),
        naive_output.data(),
        input_shape,
        kernel_shape,
        output_shape,
        stride,
        padding,
        dilation
    );

    // ライブラリ実装による forward 畳み込みの実行
    st = zenu_compute_conv_forward_cpu(
        conv_cpu,
        input_buf,
        kernel_buf,
        workspace,
        output_buf
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 結果の比較
    const float* lib_output = static_cast<const float*>(output_buf);
    ASSERT_TRUE(array_compare(lib_output, naive_output.data(), output_size, 1e-5f));

    // 後始末
    zenu_compute_free_cpu(input_buf);
    zenu_compute_free_cpu(kernel_buf);
    zenu_compute_free_cpu(output_buf);
    zenu_compute_free_cpu(workspace);
    zenu_compute_destroy_conv_cpu(conv_cpu);
}


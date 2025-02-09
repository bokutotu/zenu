#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>

#include "zenu_compute_conv.h"
#include "zenu_compute_memory.h"
#include "zenu_compute_type.h"

#include "array_comp.h"

TEST(ConvBiasNvidiaTest, Forward)
{
    size_t input_shape[4] = { 1, 3, 2, 2 };
    size_t bias_shape[1] = { 3 };

    float input_cpu[12] = {
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f
    };

    float bias_cpu[3] = { 0.5f, 1.0f, 1.5f };

    float expected_output[12] = {
         1.5f,  2.5f,  3.5f,  4.5f,
         6.0f,  7.0f,  8.0f,  9.0f,
        10.5f, 11.5f, 12.5f, 13.5f
    };

    void* d_input  = nullptr;
    void* d_bias   = nullptr;
    void* d_output = nullptr;
    int input_bytes = 12 * static_cast<int>(sizeof(float));
    int bias_bytes  = 3  * static_cast<int>(sizeof(float));

    EXPECT_EQ(zenu_compute_malloc_nvidia(&d_input, input_bytes), Success);
    EXPECT_EQ(zenu_compute_malloc_nvidia(&d_bias,  bias_bytes),  Success);
    EXPECT_EQ(zenu_compute_malloc_nvidia(&d_output, input_bytes), Success);

    EXPECT_EQ(zenu_compute_cpu_to_nvidia(d_input,  static_cast<void*>(input_cpu), input_bytes), Success);
    EXPECT_EQ(zenu_compute_cpu_to_nvidia(d_bias,   static_cast<void*>(bias_cpu),  bias_bytes),  Success);

    ZenuStatus status = zenu_compute_conv_forward_bias_nvidia(
        input_shape,
        bias_shape,
        2,
        ZenuDataType::f32,
        d_input,
        d_bias,
        d_output
    );
    EXPECT_EQ(status, Success);

    float output_cpu[12] = { 0 };
    EXPECT_EQ(zenu_compute_nvidia_to_cpu(static_cast<void*>(output_cpu), d_output, input_bytes), Success);

    bool cmp = array_compare<float>(output_cpu, expected_output, 12);
    EXPECT_TRUE(cmp);

    zenu_compute_free_nvidia(d_input);
    zenu_compute_free_nvidia(d_bias);
    zenu_compute_free_nvidia(d_output);
}

TEST(ConvBiasNvidiaTest, Backward)
{
    size_t input_shape[4] = { 1, 3, 2, 2 };
    size_t bias_shape[1] = { 3 };

    float d_output_cpu[12] = {
         1.0f,  2.0f,  3.0f,  4.0f,   // channel0
         5.0f,  6.0f,  7.0f,  8.0f,   // channel1
         9.0f, 10.0f, 11.0f, 12.0f    // channel2
    };

    float expected_d_bias[3] = { 10.0f, 26.0f, 42.0f };

    void* d_d_output = nullptr;
    void* d_d_bias   = nullptr;
    int d_output_bytes = 12 * static_cast<int>(sizeof(float));
    int d_bias_bytes   = 3  * static_cast<int>(sizeof(float));

    EXPECT_EQ(zenu_compute_malloc_nvidia(&d_d_output, d_output_bytes), Success);
    EXPECT_EQ(zenu_compute_malloc_nvidia(&d_d_bias,   d_bias_bytes),   Success);

    EXPECT_EQ(zenu_compute_cpu_to_nvidia(d_d_output, static_cast<void*>(d_output_cpu), d_output_bytes), Success);

    size_t workspace_size = 0;
    EXPECT_EQ(zenu_compute_bkwd_bias_get_workspace_nvidia(input_shape, 2, ZenuDataType::f32, &workspace_size), Success);

    // ワークスペース領域の確保 (d_workspace)
    void* d_workspace = nullptr;
    EXPECT_EQ(zenu_compute_malloc_nvidia(&d_workspace, workspace_size), Success);

    // Backward 実行 (workspace を利用)
    auto status = zenu_compute_conv_bkwd_bias_nvidia(
        input_shape,
        bias_shape,
        2,
        ZenuDataType::f32,
        d_d_output,
        d_d_bias,
        d_workspace
    );
    EXPECT_EQ(status, Success);

    float d_bias_cpu[3] = { 0 };
    EXPECT_EQ(zenu_compute_nvidia_to_cpu(static_cast<void*>(d_bias_cpu), d_d_bias, d_bias_bytes), Success);

    bool cmp = array_compare<float>(d_bias_cpu, expected_d_bias, 3);
    EXPECT_TRUE(cmp);

    zenu_compute_free_nvidia(d_d_output);
    zenu_compute_free_nvidia(d_d_bias);
    zenu_compute_free_nvidia(d_workspace);
}


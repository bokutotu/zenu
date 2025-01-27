#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "zenu_compute_conv.h"
#include "zenu_compute_memory.h"
#include "zenu_compute_random.h"
#include "zenu_compute_type.h"
#include "array_comp.h"

/**
 * @brief Naive 2D convolution implementation with NCHW layout
 * @tparam T Data type (float/double)
 * @param input Input tensor in NCHW format
 * @param kernel Convolution kernel in KCRS format
 * @param output Output tensor in NCHW format
 * @param input_shape Input shape {N, C, H, W}
 * @param kernel_shape Kernel shape {K, C, R, S}
 * @param output_shape Output shape {N, K, P, Q}
 * @param stride Stride values {stride_h, stride_w}
 * @param padding Padding values {pad_h, pad_w}
 * @param dilation Dilation values {dilation_h, dilation_w}
 */
template <typename T>
void conv2d_naive(
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
    const size_t H = input_shape[2];
    const size_t W = input_shape[3];
    
    const size_t K = kernel_shape[0];
    const size_t R = kernel_shape[2];
    const size_t S = kernel_shape[3];
    
    const size_t P = output_shape[2];
    const size_t Q = output_shape[3];

    for(size_t n = 0; n < N; ++n) {
        for(size_t k = 0; k < K; ++k) {
            for(size_t p = 0; p < P; ++p) {
                for(size_t q = 0; q < Q; ++q) {
                    T sum = 0;
                    
                    // Convolution window
                    for(size_t c = 0; c < C; ++c) {
                        for(size_t r = 0; r < R; ++r) {
                            for(size_t s = 0; s < S; ++s) {
                                // Calculate input position with dilation and padding
                                const int h = p * stride[0] - padding[0] + r * dilation[0];
                                const int w = q * stride[1] - padding[1] + s * dilation[1];
                                
                                // Boundary check
                                if(h >= 0 && h < static_cast<int>(H) && w >= 0 && w < static_cast<int>(W)) {
                                    const size_t input_idx = n * C * H * W + c * H * W + h * W + w;
                                    const size_t kernel_idx = k * C * R * S + c * R * S + r * S + s;
                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                    
                    // Store result
                    const size_t output_idx = n * K * P * Q + k * P * Q + p * Q + q;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

TEST(ZenuConvCpuTest, FloatForward2DConvTest) {
    // Initialize convolution parameters with realistic sizes
    const std::vector<size_t> input_shape = {2, 3, 14, 14};  // NCHW (batch=2, channels=3, 28x28)
    const std::vector<size_t> kernel_shape = {2, 3, 5, 5};  // KCRS (filters=16, 5x5 kernel)
    const std::vector<size_t> output_shape = {2, 2, 14, 14};// NKPQ (output spatial size 24x24)
    const std::vector<size_t> stride = {1, 1};
    const std::vector<size_t> padding = {2, 2};  // No padding
    const std::vector<size_t> dilation = {1, 1};

    // Random number generator seed
    const unsigned long long seed = 12345ULL;

    // Create convolution handle
    ZenuComputeConvCpu* conv_cpu = nullptr;
    zenu_compute_create_conv_cpu(&conv_cpu);
    
    // Set convolution descriptor
    ZenuStatus st = zenu_compute_set_conv_cpu_descriptor_cpu(
        conv_cpu,
        const_cast<size_t*>(input_shape.data()),
        const_cast<size_t*>(output_shape.data()),
        const_cast<size_t*>(kernel_shape.data()),
        const_cast<size_t*>(stride.data()),
        const_cast<size_t*>(padding.data()),
        const_cast<size_t*>(dilation.data()),
        ZenuDataType::f32,
        2
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // Allocate memory
    void *input_buf, *kernel_buf, *output_buf, *workspace;
    const size_t input_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
    const size_t kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3];
    const size_t output_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];
    
    zenu_compute_malloc_cpu(&input_buf, input_size * sizeof(float));
    zenu_compute_malloc_cpu(&kernel_buf, kernel_size * sizeof(float));
    zenu_compute_malloc_cpu(&output_buf, output_size * sizeof(float));
    
    // Get workspace size
    const size_t workspace_bytes = zenu_compute_conv_get_forward_workspace_bytes_cpu(conv_cpu);
    zenu_compute_malloc_cpu(&workspace, workspace_bytes);

    st = zenu_compute_normal_distribution_cpu(input_buf, input_size, 0., 1, f32 , 1234);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(kernel_buf, kernel_size, 0., 1, f32 , 1234);
    ASSERT_EQ(st, ZenuStatus::Success);

    // Run reference implementation
    std::vector<float> naive_output(output_size, 0);
    conv2d_naive<float>(
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

    // Run library implementation
    st = zenu_compute_conv_forward_cpu(
        conv_cpu,
        input_buf,
        kernel_buf,
        workspace,
        output_buf
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // Verify results
    const float* lib_output = static_cast<const float*>(output_buf);
    ASSERT_TRUE(array_compare(lib_output, naive_output.data(), output_size, 1e-5f));

    // Cleanup
    zenu_compute_free_cpu(input_buf);
    zenu_compute_free_cpu(kernel_buf);
    zenu_compute_free_cpu(output_buf);
    zenu_compute_free_cpu(workspace);
    zenu_compute_destroy_conv_cpu(conv_cpu);
}

/**
 * @brief Naive 2D convolution backward data implementation
 * @tparam T Data type (float/double)
 * @param d_output Output gradient tensor
 * @param kernel Convolution kernel
 * @param d_input Input gradient tensor (output)
 * @param input_shape Input shape {N, C, H, W}
 * @param kernel_shape Kernel shape {K, C, R, S}
 * @param output_shape Output shape {N, K, P, Q}
 * @param stride Stride values {stride_h, stride_w}
 * @param padding Padding values {pad_h, pad_w}
 * @param dilation Dilation values {dilation_h, dilation_w}
 */
template <typename T>
void conv2d_backward_data_naive(
    const T* d_output,
    const T* kernel,
    T* d_input,
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& kernel_shape,
    const std::vector<size_t>& output_shape,
    const std::vector<size_t>& stride,
    const std::vector<size_t>& padding,
    const std::vector<size_t>& dilation) 
{
    const size_t N = input_shape[0];
    const size_t C = input_shape[1];
    const size_t H = input_shape[2];
    const size_t W = input_shape[3];
    
    const size_t K = kernel_shape[0];
    const size_t R = kernel_shape[2];
    const size_t S = kernel_shape[3];
    
    const size_t P = output_shape[2];
    const size_t Q = output_shape[3];

    // Initialize gradients to zero
    memset(d_input, 0, N * C * H * W * sizeof(T));

    for(size_t n = 0; n < N; ++n) {
        for(size_t p = 0; p < P; ++p) {
            for(size_t q = 0; q < Q; ++q) {
                for(size_t k = 0; k < K; ++k) {
                    const T grad = d_output[n * K * P * Q + k * P * Q + p * Q + q];
                    for(size_t c = 0; c < C; ++c) {
                        for(size_t r = 0; r < R; ++r) {
                            for(size_t s = 0; s < S; ++s) {
                                // Calculate original input position
                                const int h_in = static_cast<int>(p * stride[0] + r * dilation[0] - padding[0]);
                                const int w_in = static_cast<int>(q * stride[1] + s * dilation[1] - padding[1]);
                                
                                // Accumulate gradient if position is valid
                                if(h_in >= 0 && h_in < static_cast<int>(H) && w_in >= 0 && w_in < static_cast<int>(W)) {
                                    const size_t input_idx = n * C * H * W + c * H * W + h_in * W + w_in;
                                    const size_t kernel_idx = k * C * R * S + c * R * S + r * S + s;
                                    d_input[input_idx] += grad * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(ZenuConvCpuTest, FloatBackwardData2DConvTest) {
    // Use same parameters as forward test
    const std::vector<size_t> input_shape = {2, 3, 14, 14};
    const std::vector<size_t> kernel_shape = {2, 3, 5, 5};
    const std::vector<size_t> output_shape = {2, 2, 14, 14};
    const std::vector<size_t> stride = {1, 1};
    const std::vector<size_t> padding = {2, 2};
    const std::vector<size_t> dilation = {1, 1};

    // Create convolution handle
    ZenuComputeConvCpu* conv_cpu = nullptr;
    zenu_compute_create_conv_cpu(&conv_cpu);
    
    // Set convolution descriptor (same as forward test)
    ZenuStatus st = zenu_compute_set_conv_cpu_descriptor_cpu(
        conv_cpu,
        const_cast<size_t*>(input_shape.data()),
        const_cast<size_t*>(output_shape.data()),
        const_cast<size_t*>(kernel_shape.data()),
        const_cast<size_t*>(stride.data()),
        const_cast<size_t*>(padding.data()),
        const_cast<size_t*>(dilation.data()),
        ZenuDataType::f32,
        2
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // Allocate memory
    void *d_input_buf, *kernel_buf, *d_output_buf, *workspace;
    const size_t input_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
    const size_t kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3];
    const size_t output_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];
    
    zenu_compute_malloc_cpu(&d_input_buf, input_size * sizeof(float));
    zenu_compute_malloc_cpu(&kernel_buf, kernel_size * sizeof(float));
    zenu_compute_malloc_cpu(&d_output_buf, output_size * sizeof(float));
    
    // Get workspace size for backward data
    const size_t workspace_bytes = zenu_compute_conv_get_bkwd_data_workspace_bytes_cpu(conv_cpu);
    zenu_compute_malloc_cpu(&workspace, workspace_bytes);

    // Initialize with known patterns for reproducible results
    st = zenu_compute_normal_distribution_cpu(kernel_buf, kernel_size, 0.0f, 1.0f, ZenuDataType::f32, 1234);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(d_output_buf, output_size, 0.0f, 1.0f, ZenuDataType::f32, 5678);
    ASSERT_EQ(st, ZenuStatus::Success);

    // Run reference implementation
    std::vector<float> naive_d_input(input_size, 0);
    conv2d_backward_data_naive<float>(
        static_cast<const float*>(d_output_buf),
        static_cast<const float*>(kernel_buf),
        naive_d_input.data(),
        input_shape,
        kernel_shape,
        output_shape,
        stride,
        padding,
        dilation
    );

    // Run library implementation
    st = zenu_compute_conv_backward_data_cpu(
        conv_cpu,
        d_output_buf,
        kernel_buf,
        workspace,
        d_input_buf
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // Verify results with higher tolerance due to atomic add order differences
    const float* lib_d_input = static_cast<const float*>(d_input_buf);

    ASSERT_TRUE(array_compare(lib_d_input, naive_d_input.data(), input_size, 1e-4f));

    // Cleanup
    zenu_compute_free_cpu(d_input_buf);
    zenu_compute_free_cpu(kernel_buf);
    zenu_compute_free_cpu(d_output_buf);
    zenu_compute_free_cpu(workspace);
    zenu_compute_destroy_conv_cpu(conv_cpu);
}

/**
 * @brief Naive 2D convolution backward filter implementation
 * @tparam T Data type (float/double)
 * @param input        入力テンソル (N,C,H,W)
 * @param d_output     出力勾配 (N,K,P,Q)
 * @param d_kernel     カーネル勾配 (K,C,R,S) (初期ゼロクリア済みか、ここでクリアする)
 * @param input_shape  入力形状 {N,C,H,W}
 * @param kernel_shape カーネル形状 {K,C,R,S}
 * @param output_shape 出力形状 {N,K,P,Q}
 * @param stride       {stride_h, stride_w}
 * @param padding      {pad_h, pad_w}
 * @param dilation     {dilation_h, dilation_w}
 */
template <typename T>
void conv2d_backward_kernel_naive(
    const T* input,
    const T* d_output,
    T*       d_kernel,
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& kernel_shape,
    const std::vector<size_t>& output_shape,
    const std::vector<size_t>& stride,
    const std::vector<size_t>& padding,
    const std::vector<size_t>& dilation
)
{
    const size_t N = input_shape[0];
    const size_t C = input_shape[1];
    const size_t H = input_shape[2];
    const size_t W = input_shape[3];

    const size_t K = kernel_shape[0];
    const size_t R = kernel_shape[2];
    const size_t S = kernel_shape[3];

    const size_t P = output_shape[2];
    const size_t Q = output_shape[3];

    // d_kernelをゼロ初期化（必要なら外側で memset() してもOK）
    memset(d_kernel, 0, K * C * R * S * sizeof(T));

    // 6重ループで勾配を計算
    for (size_t n = 0; n < N; ++n) {
        for (size_t k_ = 0; k_ < K; ++k_) {
            for (size_t p_ = 0; p_ < P; ++p_) {
                for (size_t q_ = 0; q_ < Q; ++q_) {
                    // 出力勾配
                    const T grad_y = d_output[n*K*P*Q + k_*P*Q + p_*Q + q_];

                    for (size_t c_ = 0; c_ < C; ++c_) {
                        for (size_t r_ = 0; r_ < R; ++r_) {
                            for (size_t s_ = 0; s_ < S; ++s_) {
                                const int h_in = static_cast<int>(p_*stride[0] + r_*dilation[0] - padding[0]);
                                const int w_in = static_cast<int>(q_*stride[1] + s_*dilation[1] - padding[1]);

                                // 入力が有効範囲に入っている場合のみ積算
                                if (h_in >= 0 && h_in < static_cast<int>(H) &&
                                    w_in >= 0 && w_in < static_cast<int>(W))
                                {
                                    const size_t in_idx = 
                                        (n*C + c_)* (H*W)
                                        + static_cast<size_t>(h_in)*W
                                        + static_cast<size_t>(w_in);

                                    const size_t w_idx  =
                                        (k_*C + c_)* (R*S)
                                        + (r_*S + s_);

                                    d_kernel[w_idx] += (grad_y * input[in_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


/**
 * @brief 2D Conv のBackward Filterテスト (float32)
 */
TEST(ZenuConvCpuTest, FloatBackwardFilter2DConvTest) {
    //------------------------------
    // 1) コンボリューションパラメータ設定
    //------------------------------
    const std::vector<size_t> input_shape  = {2, 3, 14, 14};  // N,C,H,W
    const std::vector<size_t> kernel_shape = {2, 3, 5, 5};    // K,C,R,S
    const std::vector<size_t> output_shape = {2, 2, 14, 14};  // N,K,P,Q
    const std::vector<size_t> stride       = {1, 1};
    const std::vector<size_t> padding      = {2, 2};
    const std::vector<size_t> dilation     = {1, 1};

    ZenuComputeConvCpu* conv_cpu = nullptr;
    zenu_compute_create_conv_cpu(&conv_cpu);

    zenu_compute_set_conv_cpu_descriptor_cpu(
        conv_cpu,
        const_cast<size_t*>(input_shape.data()),
        const_cast<size_t*>(output_shape.data()),
        const_cast<size_t*>(kernel_shape.data()),
        const_cast<size_t*>(stride.data()),
        const_cast<size_t*>(padding.data()),
        const_cast<size_t*>(dilation.data()),
        ZenuDataType::f32,
        2
    );

    size_t input_size  = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
    size_t kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3];
    size_t output_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];

    void* input_buf, *kernel_buf, *d_output_buf, *workspace;
    zenu_compute_malloc_cpu(&input_buf, input_size * sizeof(float));
    zenu_compute_malloc_cpu(&kernel_buf, kernel_size * sizeof(float));
    zenu_compute_malloc_cpu(&d_output_buf, output_size * sizeof(float));

    size_t workspace_bytes = zenu_compute_conv_get_bkwd_kernel_workspace_bytes_cpu(conv_cpu);
    zenu_compute_malloc_cpu(&workspace, workspace_bytes);

    zenu_compute_normal_distribution_cpu(input_buf, input_size, 0.0f, 1.0f, ZenuDataType::f32, 1234);
    zenu_compute_normal_distribution_cpu(kernel_buf, kernel_size, 0.0f, 1.0f, ZenuDataType::f32, 5678);
    zenu_compute_normal_distribution_cpu(d_output_buf, output_size, 0.0f, 1.0f, ZenuDataType::f32, 9012);

    std::vector<float> naive_d_kernel(kernel_size, 0);
    conv2d_backward_kernel_naive<float>(
        static_cast<const float*>(input_buf),
        static_cast<const float*>(d_output_buf),
        naive_d_kernel.data(),
        input_shape,
        kernel_shape,
        output_shape,
        stride,
        padding,
        dilation
    );

    ZenuStatus st = zenu_compute_conv_backward_kernel_cpu(
        conv_cpu,
        input_buf,
        d_output_buf,
        kernel_buf,
        workspace
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    const float* lib_d_kernel = static_cast<const float*>(kernel_buf);
    ASSERT_TRUE(array_compare(lib_d_kernel, naive_d_kernel.data(), kernel_size, 1e-4f));

    zenu_compute_free_cpu(input_buf);
    zenu_compute_free_cpu(kernel_buf);
    zenu_compute_free_cpu(d_output_buf);
    zenu_compute_free_cpu(workspace);
    zenu_compute_destroy_conv_cpu(conv_cpu);
}

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

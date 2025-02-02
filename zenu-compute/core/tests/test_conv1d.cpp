#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include "zenu_compute_conv.h"
#include "zenu_compute_memory.h"
#include "zenu_compute_random.h"
#include "zenu_compute_type.h"
#include "array_comp.h"

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
    const size_t N = input_shape[0], C = input_shape[1], L = input_shape[2];
    const size_t K = kernel_shape[0], S = kernel_shape[2];
    const size_t L_out = output_shape[2];
    for (size_t n = 0; n < N; ++n)
        for (size_t k = 0; k < K; ++k)
            for (size_t l = 0; l < L_out; ++l) {
                T sum = 0;
                for (size_t c = 0; c < C; ++c)
                    for (size_t s = 0; s < S; ++s) {
                        int x = static_cast<int>(l * stride[0]) - static_cast<int>(padding[0]) + static_cast<int>(s * dilation[0]);
                        if (x >= 0 && x < static_cast<int>(L))
                            sum += input[n * C * L + c * L + x] * kernel[k * C * S + c * S + s];
                    }
                output[n * K * L_out + k * L_out + l] = sum;
            }
}

template <typename T>
void conv1d_backward_data_naive(
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
    const size_t N = input_shape[0], C = input_shape[1], L = input_shape[2];
    const size_t K = kernel_shape[0], S = kernel_shape[2];
    const size_t L_out = output_shape[2];
    memset(d_input, 0, N * C * L * sizeof(T));
    for (size_t n = 0; n < N; ++n)
        for (size_t k = 0; k < K; ++k)
            for (size_t l = 0; l < L_out; ++l) {
                T grad = d_output[n * K * L_out + k * L_out + l];
                for (size_t c = 0; c < C; ++c)
                    for (size_t s = 0; s < S; ++s) {
                        int x = static_cast<int>(l * stride[0]) - static_cast<int>(padding[0]) + static_cast<int>(s * dilation[0]);
                        if (x >= 0 && x < static_cast<int>(L))
                            d_input[n * C * L + c * L + x] += grad * kernel[k * C * S + c * S + s];
                    }
            }
}

template <typename T>
void conv1d_backward_kernel_naive(
    const T* input,
    const T* grad_output,
    T* grad_kernel,
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& kernel_shape,
    const std::vector<size_t>& output_shape,
    const std::vector<size_t>& stride,
    const std::vector<size_t>& padding,
    const std::vector<size_t>& dilation)
{
    const size_t N = input_shape[0], C = input_shape[1], L = input_shape[2];
    const size_t K = kernel_shape[0], S = kernel_shape[2];
    const size_t L_out = output_shape[2];
    memset(grad_kernel, 0, K * C * S * sizeof(T));
    for (size_t n = 0; n < N; ++n)
        for (size_t k = 0; k < K; ++k)
            for (size_t l = 0; l < L_out; ++l) {
                T grad_val = grad_output[n * K * L_out + k * L_out + l];
                for (size_t c = 0; c < C; ++c)
                    for (size_t s = 0; s < S; ++s) {
                        int x = static_cast<int>(l * stride[0]) - static_cast<int>(padding[0]) + static_cast<int>(s * dilation[0]);
                        if (x >= 0 && x < static_cast<int>(L))
                            grad_kernel[k * C * S + c * S + s] += input[n * C * L + c * L + x] * grad_val;
                    }
            }
}

TEST(ZenuConvCpuTest, FloatForward1DConvTest) {
    const std::vector<size_t> input_shape  = {2, 3, 28};
    const std::vector<size_t> kernel_shape = {2, 3, 5};
    const std::vector<size_t> output_shape = {2, 2, 28};
    const std::vector<size_t> stride       = {1};
    const std::vector<size_t> padding      = {2};
    const std::vector<size_t> dilation     = {1};

    ZenuComputeConvCpu* conv_cpu = nullptr;
    ZenuStatus st = zenu_compute_create_conv_cpu(&conv_cpu);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_set_conv_cpu_descriptor_cpu(
        conv_cpu,
        const_cast<size_t*>(input_shape.data()),
        const_cast<size_t*>(output_shape.data()),
        const_cast<size_t*>(kernel_shape.data()),
        const_cast<size_t*>(stride.data()),
        const_cast<size_t*>(padding.data()),
        const_cast<size_t*>(dilation.data()),
        ZenuDataType::f32,
        1
    );
    ASSERT_EQ(st, ZenuStatus::Success);

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
    const size_t workspace_bytes = zenu_compute_conv_get_forward_workspace_bytes_cpu(conv_cpu);
    zenu_compute_malloc_cpu(&workspace, workspace_bytes);

    st = zenu_compute_normal_distribution_cpu(input_buf,  input_size, 0.0f, 1.0f, ZenuDataType::f32, 1234);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(kernel_buf, kernel_size, 0.0f, 1.0f, ZenuDataType::f32, 5678);
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<float> naive_output(output_size, 0.0f);
    conv1d_naive<float>(
        static_cast<const float*>(input_buf),
        static_cast<const float*>(kernel_buf),
        naive_output.data(),
        input_shape, kernel_shape, output_shape,
        stride, padding, dilation
    );

    st = zenu_compute_conv_forward_cpu(conv_cpu, input_buf, kernel_buf, workspace, output_buf);
    ASSERT_EQ(st, ZenuStatus::Success);
    const float* lib_output = static_cast<const float*>(output_buf);
    ASSERT_TRUE(array_compare(lib_output, naive_output.data(), output_size, 1e-5f));

    zenu_compute_free_cpu(input_buf);
    zenu_compute_free_cpu(kernel_buf);
    zenu_compute_free_cpu(output_buf);
    zenu_compute_free_cpu(workspace);
    zenu_compute_destroy_conv_cpu(conv_cpu);
}

TEST(ZenuConvCpuTest, FloatBackwardData1DConvTest) {
    const std::vector<size_t> input_shape  = {2, 3, 28};
    const std::vector<size_t> kernel_shape = {2, 3, 5};
    const std::vector<size_t> output_shape = {2, 2, 28};
    const std::vector<size_t> stride       = {1};
    const std::vector<size_t> padding      = {2};
    const std::vector<size_t> dilation     = {1};

    ZenuComputeConvCpu* conv_cpu = nullptr;
    ZenuStatus st = zenu_compute_create_conv_cpu(&conv_cpu);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_set_conv_cpu_descriptor_cpu(
        conv_cpu,
        const_cast<size_t*>(input_shape.data()),
        const_cast<size_t*>(output_shape.data()),
        const_cast<size_t*>(kernel_shape.data()),
        const_cast<size_t*>(stride.data()),
        const_cast<size_t*>(padding.data()),
        const_cast<size_t*>(dilation.data()),
        ZenuDataType::f32,
        1
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    const size_t input_size  = input_shape[0] * input_shape[1] * input_shape[2];
    const size_t kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
    const size_t output_size = output_shape[0] * output_shape[1] * output_shape[2];

    void* d_input_buf  = nullptr;
    void* kernel_buf   = nullptr;
    void* d_output_buf = nullptr;
    void* workspace    = nullptr;
    zenu_compute_malloc_cpu(&d_input_buf,  input_size  * sizeof(float));
    zenu_compute_malloc_cpu(&kernel_buf,   kernel_size * sizeof(float));
    zenu_compute_malloc_cpu(&d_output_buf, output_size * sizeof(float));
    const size_t workspace_bytes = zenu_compute_conv_get_bkwd_data_workspace_bytes_cpu(conv_cpu);
    zenu_compute_malloc_cpu(&workspace, workspace_bytes);

    st = zenu_compute_normal_distribution_cpu(kernel_buf, kernel_size, 0.0f, 1.0f, ZenuDataType::f32, 1234);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(d_output_buf, output_size, 0.0f, 1.0f, ZenuDataType::f32, 5678);
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<float> naive_d_input(input_size, 0.0f);
    conv1d_backward_data_naive<float>(
        static_cast<const float*>(d_output_buf),
        static_cast<const float*>(kernel_buf),
        naive_d_input.data(),
        input_shape, kernel_shape, output_shape,
        stride, padding, dilation
    );

    st = zenu_compute_conv_backward_data_cpu(conv_cpu, d_output_buf, kernel_buf, workspace, d_input_buf);
    ASSERT_EQ(st, ZenuStatus::Success);
    const float* lib_d_input = static_cast<const float*>(d_input_buf);
    ASSERT_TRUE(array_compare(lib_d_input, naive_d_input.data(), input_size, 1e-4f));

    zenu_compute_free_cpu(d_input_buf);
    zenu_compute_free_cpu(kernel_buf);
    zenu_compute_free_cpu(d_output_buf);
    zenu_compute_free_cpu(workspace);
    zenu_compute_destroy_conv_cpu(conv_cpu);
}

TEST(ZenuConvCpuTest, FloatBackwardKernel1DConvTest) {
    const std::vector<size_t> input_shape  = {2, 3, 28};
    const std::vector<size_t> kernel_shape = {2, 3, 5};
    const std::vector<size_t> output_shape = {2, 2, 28};
    const std::vector<size_t> stride       = {1};
    const std::vector<size_t> padding      = {2};
    const std::vector<size_t> dilation     = {1};

    ZenuComputeConvCpu* conv_cpu = nullptr;
    ZenuStatus st = zenu_compute_create_conv_cpu(&conv_cpu);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_set_conv_cpu_descriptor_cpu(
        conv_cpu,
        const_cast<size_t*>(input_shape.data()),
        const_cast<size_t*>(output_shape.data()),
        const_cast<size_t*>(kernel_shape.data()),
        const_cast<size_t*>(stride.data()),
        const_cast<size_t*>(padding.data()),
        const_cast<size_t*>(dilation.data()),
        ZenuDataType::f32,
        1
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    const size_t input_size       = input_shape[0] * input_shape[1] * input_shape[2];
    const size_t grad_output_size = output_shape[0] * output_shape[1] * output_shape[2];
    const size_t grad_kernel_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];

    void* input_buf       = nullptr;
    void* grad_output_buf = nullptr;
    void* grad_kernel_buf = nullptr;
    void* workspace       = nullptr;
    zenu_compute_malloc_cpu(&input_buf,        input_size * sizeof(float));
    zenu_compute_malloc_cpu(&grad_output_buf,  grad_output_size * sizeof(float));
    zenu_compute_malloc_cpu(&grad_kernel_buf,  grad_kernel_size * sizeof(float));
    const size_t ws_bytes = zenu_compute_conv_get_bkwd_kernel_workspace_bytes_cpu(conv_cpu);
    zenu_compute_malloc_cpu(&workspace, ws_bytes);

    st = zenu_compute_normal_distribution_cpu(input_buf, input_size, 0.0f, 1.0f, ZenuDataType::f32, 1234);
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_normal_distribution_cpu(grad_output_buf, grad_output_size, 0.0f, 1.0f, ZenuDataType::f32, 5678);
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<float> naive_grad_kernel(grad_kernel_size, 0.0f);
    conv1d_backward_kernel_naive<float>(
        static_cast<const float*>(input_buf),
        static_cast<const float*>(grad_output_buf),
        naive_grad_kernel.data(),
        input_shape, kernel_shape, output_shape,
        stride, padding, dilation
    );

    st = zenu_compute_conv_backward_kernel_cpu(conv_cpu, input_buf, grad_output_buf, grad_kernel_buf, workspace);
    ASSERT_EQ(st, ZenuStatus::Success);
    const float* lib_grad_kernel = static_cast<const float*>(grad_kernel_buf);
    ASSERT_TRUE(array_compare(lib_grad_kernel, naive_grad_kernel.data(), grad_kernel_size, 1e-4f));

    zenu_compute_free_cpu(input_buf);
    zenu_compute_free_cpu(grad_output_buf);
    zenu_compute_free_cpu(grad_kernel_buf);
    zenu_compute_free_cpu(workspace);
    zenu_compute_destroy_conv_cpu(conv_cpu);
}


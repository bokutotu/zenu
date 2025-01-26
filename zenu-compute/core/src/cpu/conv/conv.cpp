#include "zenu_compute_conv.h"
#include "zenu_compute_type.h"
#include "conv.h"

struct ZenuComputeConvCpu {
    ZenuComputeConvCpuImpl* impl;
};

ZenuStatus zenu_compute_create_conv_cpu(ZenuComputeConvCpu** conv_cpu) {
    if (conv_cpu == nullptr) {
        return InvalidArgument;
    }

    *conv_cpu = new ZenuComputeConvCpu();
    (*conv_cpu)->impl = new ZenuComputeConvCpuImpl();
    return Success;
}

void zenu_compute_destroy_conv_cpu(ZenuComputeConvCpu* conv_cpu) {
    delete conv_cpu;
}

ZenuStatus zenu_compute_set_conv_cpu_descriptor_cpu(
    ZenuComputeConvCpu* conv_cpu,
    size_t* input,
    size_t* output,
    size_t* kernel,
    size_t* stride,
    size_t* padding,
    size_t* dilation,
    ZenuDataType type,
    size_t num_dim
) {
    auto input_vec = std::vector<size_t>(input, input + num_dim + 2);
    auto output_vec = std::vector<size_t>(output, output + num_dim + 2);
    auto kernel_vec = std::vector<size_t>(kernel, kernel + num_dim + 2);
    auto stride_vec = std::vector<size_t>(stride, stride + num_dim);
    auto padding_vec = std::vector<size_t>(padding, padding + num_dim);
    auto dilation_vec = std::vector<size_t>(dilation, dilation + num_dim);
    return conv_cpu->impl->init(input_vec, output_vec, kernel_vec, stride_vec, padding_vec, dilation_vec, type);
}

size_t zenu_compute_conv_get_forward_workspace_bytes_cpu(ZenuComputeConvCpu* conv_cpu) {
    return conv_cpu->impl->get_forward_bytes();
}

ZenuStatus zenu_compute_conv_forward_cpu(ZenuComputeConvCpu* conv_cpu, 
                                         const void* input_buf, 
                                         const void*kernel_buf, 
                                         void* workspace, 
                                         void* output) {
    return conv_cpu->impl->forward(input_buf, kernel_buf, output, workspace);
}

size_t zenu_compute_conv_get_bkwd_data_workspace_bytes_cpu(ZenuComputeConvCpu* conv_cpu) {
    return conv_cpu->impl->get_backward_data_bytes();
}

ZenuStatus zenu_compute_conv_backward_data_cpu(
    ZenuComputeConvCpu* conv_cpu,
    const void* d_output,
    const void* kernel,
    void* workspace,
    void* d_input
) {
    return conv_cpu->impl->backward_data(kernel, d_output, d_input, workspace);
}

size_t zenu_compute_conv_get_bkwd_kernel_workspace_bytes_cpu(ZenuComputeConvCpu* conv_cpu) {
    return conv_cpu->impl->get_backward_kernel_bytes();
}

ZenuStatus zenu_compute_conv_backward_kernel_cpu(
    ZenuComputeConvCpu* conv_cpu,
    const void* input,
    const void* grad_output,
    void* grad_kernel,
    void* workspace
) {
    return conv_cpu->impl->backward_kernel(input, grad_output, grad_kernel, workspace);
}

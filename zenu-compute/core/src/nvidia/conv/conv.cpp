#include "conv_interface.h"
#include "zenu_compute_conv.h"
#include "zenu_compute_type.h"

struct ZenuComputeConvNvidia {
    ZenuComputeConvNvidiaImpl* conv_nvidia;
};

ZenuStatus zenu_compute_create_conv_nvidia(ZenuComputeConvNvidia** conv_nvidia) {
    if (conv_nvidia == nullptr) {
        return InvalidArgument;
    }

    *conv_nvidia = new ZenuComputeConvNvidia();
    (*conv_nvidia)->conv_nvidia = new ZenuComputeConvNvidiaImpl();
    return Success;
}

void zenu_compute_destroy_conv_nvidia(ZenuComputeConvNvidia* conv_nvidia) {
    delete conv_nvidia;
}

ZenuStatus zenu_compute_set_conv_nvidia_descriptor(
    ZenuComputeConvNvidia* conv_nvidia,
    size_t* input,
    size_t* output,
    size_t* kernel,
    size_t* padding,
    size_t* stride,
    size_t* dilation,
    ZenuDataType type,
    size_t num_dim
) {
    auto input_vec = std::vector<size_t>(input, input + num_dim + 2);
    auto output_vec = std::vector<size_t>(output, output + num_dim + 2);
    auto kernel_vec = std::vector<size_t>(kernel, kernel + num_dim + 2);
    auto padding_vec = std::vector<size_t>(padding, padding + num_dim);
    auto stride_vec = std::vector<size_t>(stride, stride + num_dim);
    auto dilation_vec = std::vector<size_t>(dilation, dilation + num_dim);
    return conv_nvidia->conv_nvidia->init(input_vec, 
                                          output_vec, 
                                          kernel_vec, 
                                          stride_vec, 
                                          padding_vec, 
                                          dilation_vec, 
                                          type);
}

size_t zenu_compute_conv_get_forward_workspace_bytes_nvidia(ZenuComputeConvNvidia* conv_nvidia) {
    return conv_nvidia->conv_nvidia->get_forward_bytes();
}

ZenuStatus zenu_compute_conv_forward_nvidia(
    ZenuComputeConvNvidia* conv_nvidia,
    const void* input,
    const void* kernel,
    void* workspace,
    void* output
) {
    return conv_nvidia->conv_nvidia->forward(input, kernel, output, workspace);
}

size_t zenu_compute_conv_get_bkwd_data_workspace_bytes_nvidia(ZenuComputeConvNvidia* conv_nvidia) {
    return conv_nvidia->conv_nvidia->get_backward_data_bytes();
}

ZenuStatus zenu_compute_conv_backward_data_nvidia(
    ZenuComputeConvNvidia* conv_nvidia,
    const void* d_output,
    const void* kernel,
    void* workspace,
    void* d_input
) {
    return conv_nvidia->conv_nvidia->backward_data(kernel, d_output, d_input, workspace);
}

size_t zenu_compute_conv_get_bkwd_kernel_workspace_bytes_nvidia(ZenuComputeConvNvidia* conv_nvidia) {
    return conv_nvidia->conv_nvidia->get_backward_kernel_bytes();
}

ZenuStatus zenu_compute_conv_backward_kernel_nvidia(
    ZenuComputeConvNvidia* conv_nvidia,
    const void* d_output,
    const void* input,
    void* workspace,
    void* d_kernel
) {
    return conv_nvidia->conv_nvidia->backward_kernel(input, d_output, d_kernel, workspace);
}

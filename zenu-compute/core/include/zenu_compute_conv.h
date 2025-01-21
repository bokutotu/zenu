#pragma once

#include "zenu_compute_type.h"
#include <cstddef>

typedef struct ZenuComputeConvCpu ZenuComputeConvCpu;

ZenuStatus zenu_compute_init_conv_cpu(ZenuComputeConvCpu** conv_cpu,
                                      size_t batch_size,
                                      size_t channels_in,
                                      size_t channels_out,
                                      size_t input_shape[2],
                                      size_t output_shape[2],
                                      size_t kernel_shape[2],
                                      size_t pad[2],
                                      size_t stride[2],
                                      size_t dilation[2],
                                      ZenuDataType data_type,
                                      size_t num_dim);

size_t zenu_compute_conv_get_forward_workspace_bytes(ZenuComputeConvCpu* conv_cpu);

ZenuStatus zenu_compute_conv_forward(ZenuComputeConvCpu* conv_cpu,
                                     const void* input,
                                     const void* kernel,
                                     void* workspace,
                                     void* output);

size_t zenu_compute_conv_get_bkwd_data_workspace_bytes(ZenuComputeConvCpu* conv_cpu);

ZenuStatus zenu_compute_conv_backward_data(ZenuComputeConvCpu* conv_cpu,
                                           const void* d_output,
                                           const void* kernel,
                                           void* workspace,
                                           void* d_input);

size_t zenu_compute_conv_get_bkwd_filter_workspace_bytes(ZenuComputeConvCpu* conv_cpu);

ZenuStatus zenu_compute_conv_backward_filter(ZenuComputeConvCpu* conv_cpu,
                                             const void* d_output,
                                             const void* input,
                                             void* workspace,
                                             void* d_kernel);

void zenu_compute_destroy_conv_cpu(ZenuComputeConvCpu* conv_cpu);

typedef struct ZenuComputeConvNvidia ZenuComputeConvNvidia;

ZenuStatus zenu_compute_init_conv_nvidia(ZenuComputeConvNvidia** conv_nvidia,
                                      size_t batch_size,
                                      size_t channels_in,
                                      size_t channels_out,
                                      size_t input_shape[2],
                                      size_t output_shape[2],
                                      size_t kernel_shape[2],
                                      size_t pad[2],
                                      size_t stride[2],
                                      size_t dilation[2],
                                      ZenuDataType data_type,
                                      size_t num_dim);

size_t zenu_compute_conv_get_forward_workspace_bytes(ZenuComputeConvNvidia* conv_nvidia);

ZenuStatus zenu_compute_conv_forward(ZenuComputeConvNvidia* conv_nvidia,
                                     const void* input,
                                     const void* kernel,
                                     void* workspace,
                                     void* output);

size_t zenu_compute_conv_get_bkwd_data_workspace_bytes(ZenuComputeConvNvidia* conv_nvidia);

ZenuStatus zenu_compute_conv_backward_data(ZenuComputeConvNvidia* conv_nvidia,
                                           const void* d_output,
                                           const void* kernel,
                                           void* workspace,
                                           void* d_input);

size_t zenu_compute_conv_get_bkwd_filter_workspace_bytes(ZenuComputeConvNvidia* conv_nvidia);

ZenuStatus zenu_compute_conv_backward_filter(ZenuComputeConvNvidia* conv_nvidia,
                                             const void* d_output,
                                             const void* input,
                                             void* workspace,
                                             void* d_kernel);

void zenu_compute_destroy_conv_nvidia(ZenuComputeConvNvidia* conv_nvidia);

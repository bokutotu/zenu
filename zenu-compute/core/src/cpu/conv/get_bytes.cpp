#include "conv.h"

#include <cstddef>
#include <iostream>

size_t ZenuComputeConvCpu::get_im2col2d_bytes() const {
    const size_t N           = this->input[0];
    const size_t C           = this->input[1];
    const size_t H           = this->input[2];
    const size_t W           = this->input[3];

    const size_t kernel_h    = this->kernel[2];
    const size_t kernel_w    = this->kernel[3];

    const size_t stride_h    = this->stride[0];
    const size_t stride_w    = this->stride[1];
    const size_t pad_h       = this->padding[0];
    const size_t pad_w       = this->padding[1];
    const size_t dilation_h  = this->dilation[0];
    const size_t dilation_w  = this->dilation[1];

    const size_t out_h       = this->output[2];
    const size_t out_w       = this->output[3];

    size_t data_size;
    switch (type) {
    case ZenuDataType::f32:
        data_size = sizeof(float);
        break;
    case ZenuDataType::f64:
        data_size = sizeof(double);
        break;
    default:
        std::cout << "Unsupported data type" << std::endl;
        exit(1);
    }

    return N * C * kernel_h * kernel_w * out_h * out_w * data_size;
}

size_t ZenuComputeConvCpu::get_im2col_bytes() const {
    if (get_dim() == 2) {
        return get_im2col2d_bytes();
    } else {
        return 0;
    }
}

size_t ZenuComputeConvCpu::get_gemm_bytes() const {
    size_t num_elm = 1;
    for (int i = 0; i < output.size(); i++) {
        num_elm *= output[i];
    }
    size_t data_bytes;
    switch (type) {
    case ZenuDataType::f32:
        data_bytes = sizeof(float);
        break;
    case ZenuDataType::f64:
        data_bytes = sizeof(double);
        break;
    default:
        std::cout << "Unsupported data type" << std::endl;
        exit(1);
    }
    return num_elm * data_bytes;
}

size_t ZenuComputeConvCpu::get_forward_output_bytes() const {
    return get_im2col_bytes() + get_gemm_bytes() + 1024;
}

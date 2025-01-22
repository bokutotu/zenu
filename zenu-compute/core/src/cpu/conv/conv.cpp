#include "conv.h"
#include <iostream>

ZenuStatus ZenuComputeConvCpuImpl::init(std::vector<size_t> input, 
                                    std::vector<size_t> output, 
                                    std::vector<size_t> kernel, 
                                    std::vector<size_t> stride, 
                                    std::vector<size_t> padding, 
                                    std::vector<size_t> dilation,
                                    ZenuDataType type) {
    this->input = input;
    this->output = output;
    this->kernel = kernel;
    this->stride = stride;
    this->padding = padding;
    this->dilation = dilation;
    this->type = type;

    if (input.size() != output.size() || input.size() != kernel.size()) {
        return InvalidArgument;
    }
    if (stride.size() != input.size() - 2 || padding.size() != input.size() - 2 || dilation.size() != input.size() - 2) {
        return InvalidArgument;
    }
    return Success;
}

size_t ZenuComputeConvCpuImpl::get_dim() const {
    return input.size() - 2;
}

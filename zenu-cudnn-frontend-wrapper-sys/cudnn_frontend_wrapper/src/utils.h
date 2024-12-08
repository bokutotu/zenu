#pragma once

#include <vector>
#include <sstream>

#include "cudnn_frontend.h"
#include "cudnn_frontend_wrapper.h"

std::vector<int64_t> from_shape(size_t n, int64_t dims[8]);

cudnn_frontend::DataType_t get_data_type(CudnnFrontendDataType_t data_type);

cudnn_frontend::graph::Tensor_attributes get_tensor_attributes(std::vector<int64_t> shape,
                                                               std::vector<int64_t> strides,
                                                               CudnnFrontendDataType_t data_type);

cudnn_frontend::graph::Tensor_attributes get_tensor_attributes_without_type(std::vector<int64_t> shape,
                                                               std::vector<int64_t> strides);

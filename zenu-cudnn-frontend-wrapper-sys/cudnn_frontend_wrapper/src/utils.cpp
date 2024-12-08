#include "cudnn_frontend.h"
#include "cudnn_frontend_wrapper.h"
#include "utils.h"

#include <vector>
#include <sstream>

std::vector<int64_t> from_shape(size_t n, int64_t dims[8]) {
    std::vector<int64_t> shape;
    for (size_t i = 0; i < n; i++) {
        shape.push_back(dims[i]);
    }
    return shape;
}

cudnn_frontend::DataType_t get_data_type(CudnnFrontendDataType_t data_type) {
    switch (data_type) {
        case DATA_TYPE_HALF:
            return cudnn_frontend::DataType_t::HALF;
        case DATA_TYPE_FLOAT:
            return cudnn_frontend::DataType_t::FLOAT;
        case DATA_TYPE_DOUBLE:
            return cudnn_frontend::DataType_t::DOUBLE;
        default:
            std::stringstream err_msg;
            err_msg << "Invalid data type: " << data_type;
            throw std::runtime_error(err_msg.str());
    }
}

cudnn_frontend::graph::Tensor_attributes get_tensor_attributes(std::vector<int64_t> shape,
                                                               std::vector<int64_t> strides,
                                                               CudnnFrontendDataType_t data_type) {
    auto type = get_data_type(data_type);
    return cudnn_frontend::graph::Tensor_attributes()
        .set_dim(shape)
        .set_stride(strides)
        .set_data_type(type);
}

cudnn_frontend::graph::Tensor_attributes get_tensor_attributes_without_type(std::vector<int64_t> shape,
                                                               std::vector<int64_t> strides) {
    return cudnn_frontend::graph::Tensor_attributes()
        .set_dim(shape)
        .set_stride(strides);
}

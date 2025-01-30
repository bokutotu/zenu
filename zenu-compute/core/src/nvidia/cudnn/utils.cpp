#include "utils.h"
#include "nvidia/handle.h"
#include "zenu_compute_type.h"

#include <sstream>

std::vector<int64_t> convert(std::vector<size_t> v) {
    std::vector<int64_t> result;
    for (size_t i = 0; i < v.size(); i++) {
        result.push_back(v[i]);
    }
    return result;
}

std::vector<int64_t> default_stride(std::vector<size_t> shape) {
    auto shape_ = convert(shape);
    std::reverse(shape_.begin(), shape_.end());
    std::vector<int64_t> strides = {1};
    for (size_t i = 0; i < shape_.size() - 1; i++) {
        strides.push_back(strides[i] * shape_[i]);
    }
    std::reverse(strides.begin(), strides.end());
    return strides;
}

cudnn_frontend::DataType_t get_data_type(ZenuDataType data_type) {
    switch (data_type) {
        case f32:
            return cudnn_frontend::DataType_t::FLOAT;
        case f64:
            return cudnn_frontend::DataType_t::DOUBLE;
        default: {
            std::stringstream err_msg;
            err_msg << "Invalid data type: " << data_type;
            exit(1);
        }
    }
}

cudnn_frontend::graph::Tensor_attributes get_tensor_attributes(std::vector<size_t> shape,
                                                               ZenuDataType data_type) {
    auto type = get_data_type(data_type);
    auto stride = default_stride(shape);

    return cudnn_frontend::graph::Tensor_attributes()
        .set_dim(convert(shape))
        .set_stride(default_stride(shape))
        .set_data_type(type);
}

cudnn_frontend::graph::Tensor_attributes get_tensor_attributes_without_type(std::vector<size_t> shape) {
    return cudnn_frontend::graph::Tensor_attributes()
        .set_dim(convert(shape))
        .set_stride(default_stride(shape));
}

ZenuStatus build_and_check_graph(cudnn_frontend::graph::Graph& graph, std::vector<fe::HeurMode_t> mode) {
    cudnnHandle_t handle = NvidiaHandles::getCudnnHandle();
    auto err = graph.validate();
    if (!err.is_good()) {
        return CudnnError;
    }

    err = graph.build_operation_graph(handle);
    if (!err.is_good()) {
        return CudnnError;
    }

    err = graph.create_execution_plans(mode);
    if (!err.is_good()) {
        return CudnnError;
    }

    err = graph.check_support(handle);
    if (!err.is_good()) {
        return CudnnError;
    }

    err = graph.build_plans(handle, cudnn_frontend::BuildPlanPolicy_t::ALL);
    if (!err.is_good()) {
        return CudnnError;
    }

    return Success;
}

size_t get_workspace_size(cudnn_frontend::graph::Graph& graph) {
    int64_t workspace_size = 0;
    auto err = graph.get_workspace_size(workspace_size);
    if (!err.is_good()) {
        return CudnnError;
    }
    return workspace_size;
}

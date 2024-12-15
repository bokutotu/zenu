#include "batchnorm.h"
#include "utils.h"
#include <vector>

static std::string type_to_string(cudnn_frontend::DataType_t type) {
    switch (type) {
        case cudnn_frontend::DataType_t::HALF:
            return "half";
        case cudnn_frontend::DataType_t::FLOAT:
            return "float";
        case cudnn_frontend::DataType_t::DOUBLE:
            return "double";
        default:
            return "unknown";
    }
}

static std::vector<int64_t> get_stat_shape(size_t n, int64_t dims[8]) {
    std::vector<int64_t> shape = from_shape(n, dims);
    if (shape.size() == 4) {
        shape[0] = 1;
        shape[2] = 1;
        shape[3] = 1;
        return shape; 
    } else {
        throw std::runtime_error("Invalid shape for stats (only supports BN1D or BN2D)");
    }
}

static std::vector<int64_t> get_stat_stride(std::vector<int64_t> shape) {
    if (shape.size() == 4) {
        std::vector<int64_t> stride(4, 1);
        stride[0] = shape[1];
        stride[1] = 1;
        stride[2] = 1;
        stride[3] = 1;
        return stride;
    } else {
        throw std::runtime_error("Invalid shape for scale/bias (only supports BN1D or BN2D)");
    }
}

static std::vector<int64_t> get_peer_stats_shape(size_t n, int64_t dims[8]) {
    std::vector<int64_t> shape = from_shape(n, dims);
    if (shape.size() == 4) {
        shape[0] = 2;
        shape[1] *= 4;
        shape[2] = 1;
        shape[3] = 1;
        return shape; 
    } else {
        throw std::runtime_error("Invalid shape for peer stats (only supports BN1D or BN2D)");
    }
}

static std::vector<int64_t> get_peer_stats_stride(std::vector<int64_t> shape) {
    if (shape.size() == 4) {
        std::vector<int64_t> stride(4, 1);
        stride[0] = shape[1];
        stride[1] = 1;
        stride[2] = 1;
        stride[3] = 1;
        return stride;
    } else {
        throw std::runtime_error("Invalid shape for peer stats (only supports BN1D or BN2D)");
    }
}


static void debug_print_(std::shared_ptr<fe::graph::Tensor_attributes> tensor) {
    std::cout << "Tensor: " << tensor->get_name() << std::endl;
    std::cout << "Shape: ";
    for (auto dim : tensor->get_dim()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    std::cout << "Strides: ";
    for (auto stride : tensor->get_stride()) {
        std::cout << stride << " ";
    }
    std::cout << std::endl;
    auto type = tensor->get_data_type();
    auto type_string = type_to_string(type);
    std::cout << "Data type: " << type_string << std::endl;
}

void BatchNormTensorAttributes::debug_print() {
    std::cout << "X ";
    debug_print_(X);
    std::cout << "prev_running_mean ";
    debug_print_(prev_running_mean);
    std::cout << "prev_running_var ";
    debug_print_(prev_running_var);
    std::cout << "scale ";
    debug_print_(scale);
    std::cout << "bias ";
    debug_print_(bias);
    std::cout << "peer_stats_0 ";
    debug_print_(peer_stats_0);
    std::cout << "peer_stats_1 ";
    debug_print_(peer_stats_1);
    std::cout << "epsilon ";
    debug_print_(epsilon);
    std::cout << "momentum ";
    debug_print_(momentum);
    std::cout << "next_running_mean ";
    debug_print_(next_running_mean);
    std::cout << "next_running_var ";
    debug_print_(next_running_var);
    std::cout << "bn_output ";
    debug_print_(bn_output);
    std::cout << "mean ";
    debug_print_(mean);
    std::cout << "inv_variance ";
    debug_print_(inv_variance);
}

BatchNormTensorAttributes::BatchNormTensorAttributes(CudnnTensorShapeStride input_shape, 
                                                     fe::graph::Graph &graph,
                                                     CudnnFrontendDataType_t type, 
                                                     bool has_running_stats,
                                                     float epsilon_val,
                                                     float momentum_val) {
    std::vector<int64_t> x_shape = from_shape(input_shape.num_dims, input_shape.dims);
    std::vector<int64_t> x_strides = from_shape(input_shape.num_dims, input_shape.strides);

    std::vector<int64_t> stat_shape = get_stat_shape(input_shape.num_dims, input_shape.dims);
    std::vector<int64_t> stat_strides = get_stat_stride(stat_shape);

    std::vector<int64_t> peer_stats_shape = get_peer_stats_shape(input_shape.num_dims, input_shape.dims);
    std::vector<int64_t> peer_stats_strides = get_peer_stats_stride(peer_stats_shape);

    X = graph.tensor(get_tensor_attributes(x_shape, x_strides, type));
    prev_running_mean = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    prev_running_var = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    scale = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    bias = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    peer_stats_0 = graph.tensor(get_tensor_attributes(peer_stats_shape, peer_stats_strides, type));
    peer_stats_1 = graph.tensor(get_tensor_attributes(peer_stats_shape, peer_stats_strides, type));
    this->epsilon = graph.tensor(fe::graph::Tensor_attributes(epsilon_val));
    this->momentum = graph.tensor(fe::graph::Tensor_attributes(momentum_val));
    auto batchnorm_options = 
        fe::graph::Batchnorm_attributes()
            .set_epsilon(this->epsilon)
            .set_peer_stats({peer_stats_0, peer_stats_1});
    if (has_running_stats) {
        batchnorm_options
            .set_previous_running_stats(prev_running_mean, prev_running_var, this->momentum);
    }
    auto [bn_output, mean, inv_variance, next_running_mean, next_running_var] = 
        graph.batchnorm(X, scale, bias, batchnorm_options);
    auto data_type = get_data_type(type);
    mean->set_output(true).set_data_type(data_type).set_dim(stat_shape).set_stride(stat_strides);
    inv_variance->set_output(true).set_data_type(data_type).set_dim(stat_shape).set_stride(stat_strides);
    if (has_running_stats) {
        next_running_mean->set_output(true).set_data_type(data_type).set_dim(stat_shape).set_stride(stat_strides);
        next_running_var->set_output(true).set_data_type(data_type).set_dim(stat_shape).set_stride(stat_strides);
    }
    bn_output->set_output(true).set_data_type(data_type).set_dim(x_shape).set_stride(x_strides);
    this->bn_output = bn_output;
    this->mean = mean;
    this->inv_variance = inv_variance;
    this->next_running_mean = next_running_mean;
    this->next_running_var = next_running_var;
}

BatchNormDescriptor::BatchNormDescriptor(CudnnTensorShapeStride input_shape_stride, 
                                         bool has_running_stats,
                                         CudnnFrontendDataType_t type,
                                         float epsilon,
                                         float momentum) :
    has_running_stats(has_running_stats) {
    auto data_type = get_data_type(type);
    graph.set_io_data_type(data_type)
        .set_intermediate_data_type(data_type)
        .set_compute_data_type(data_type);

    attributes = BatchNormTensorAttributes(input_shape_stride, graph, type, has_running_stats, epsilon, momentum);
}

CudnnFrontendError_t BatchNormDescriptor::check_graph(cudnnHandle_t* handle) {
    return build_and_check_graph(handle, false);
}

CudnnFrontendError_t BatchNormDescriptor::get_workspace_size(int64_t* workspace_size) {
    return IGraphDescriptor::get_workspace_size(workspace_size);
}

CudnnFrontendError_t BatchNormDescriptor::execute(cudnnHandle_t* handle, 
                                                  BatchNormExecutionBuffers* buffers, 
                                                  void* workspace) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {attributes.X, buffers->X},
        {attributes.mean, buffers->mean},
        {attributes.inv_variance, buffers->inv_variance},
        {attributes.scale, buffers->scale},
        {attributes.bias, buffers->bias},
        {attributes.bn_output, buffers->Y},
        {attributes.peer_stats_0, buffers->peer_stats_0},
        {attributes.peer_stats_1, buffers->peer_stats_1}};

    if (has_running_stats) {
        variant_pack[attributes.prev_running_mean] = buffers->prev_running_mean;
        variant_pack[attributes.prev_running_var]  = buffers->prev_running_var;
        variant_pack[attributes.next_running_mean] = buffers->next_running_mean;
        variant_pack[attributes.next_running_var]  = buffers->next_running_var;
    }

    return execute_graph(handle, variant_pack, workspace);
}

BatchNormBkwdTensorAttributes::BatchNormBkwdTensorAttributes(CudnnTensorShapeStride input_shape, 
                                                             fe::graph::Graph &graph, 
                                                             CudnnFrontendDataType_t type) {
    std::vector<int64_t> x_shape = from_shape(input_shape.num_dims, input_shape.dims);
    std::vector<int64_t> x_strides = from_shape(input_shape.num_dims, input_shape.strides);

    std::vector<int64_t> stat_shape = get_stat_shape(input_shape.num_dims, input_shape.dims);
    std::vector<int64_t> stat_strides = get_stat_stride(stat_shape);

    std::vector<int64_t> peer_stats_shape = get_peer_stats_shape(input_shape.num_dims, 
                                                                 input_shape.dims);
    std::vector<int64_t> peer_stats_strides = get_peer_stats_stride(peer_stats_shape);

    DY = graph.tensor(get_tensor_attributes(x_shape, x_strides, type));
    X = graph.tensor(get_tensor_attributes(x_shape, x_strides, type));
    scale = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    mean = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    inv_variance = graph.tensor(get_tensor_attributes(stat_shape, stat_strides, type));
    peer_stats_0 = graph.tensor(get_tensor_attributes(peer_stats_shape, peer_stats_strides, type));
    peer_stats_1 = graph.tensor(get_tensor_attributes(peer_stats_shape, peer_stats_strides, type));

    auto dbn_options = fe::graph::Batchnorm_backward_attributes()
                            .set_saved_mean_and_inv_variance(mean, inv_variance)
                            .set_peer_stats({peer_stats_0, peer_stats_1});

    auto [DX, dscale, dbias] = graph.batchnorm_backward(DY, X, scale, dbn_options);
    DX->set_output(true);
    dscale->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    dbias->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    this->DX = DX;
    this->dscale = dscale;
    this->dbias = dbias;
}

BatchNormBkwdDescriptor::BatchNormBkwdDescriptor(CudnnTensorShapeStride input_shape_stride, 
                                                 CudnnFrontendDataType_t type) {
    auto data_type = get_data_type(type);
    graph.set_io_data_type(data_type)
         .set_intermediate_data_type(data_type)
         .set_compute_data_type(data_type);

    attributes = BatchNormBkwdTensorAttributes(input_shape_stride, graph, type);
}

CudnnFrontendError_t BatchNormBkwdDescriptor::check_graph(cudnnHandle_t* handle) {
    return build_and_check_graph(handle, true);
}

CudnnFrontendError_t BatchNormBkwdDescriptor::get_workspace_size(int64_t* workspace_size) {
    return IGraphDescriptor::get_workspace_size(workspace_size);
}

CudnnFrontendError_t BatchNormBkwdDescriptor::execute(cudnnHandle_t* handle, 
                                                      BatchNormBkwdExecutionBuffers* buffers,
                                                      void* workspace) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {attributes.X, buffers->X},
        {attributes.DY, buffers->DY},
        {attributes.mean, buffers->mean},
        {attributes.inv_variance, buffers->inv_variance},
        {attributes.scale, buffers->scale},
        {attributes.dscale, buffers->dscale},
        {attributes.dbias, buffers->dbias},
        {attributes.DX, buffers->DX},
        {attributes.peer_stats_0, buffers->peer_stats_0},
        {attributes.peer_stats_1, buffers->peer_stats_1}};

    return execute_graph(handle, variant_pack, workspace);
}

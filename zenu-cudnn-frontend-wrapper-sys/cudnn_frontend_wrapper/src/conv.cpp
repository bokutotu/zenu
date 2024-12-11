#include "conv.h"
#include "utils.h"
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

void debug_shape_stride(CudnnTensorShapeStride* shape) {
    std::cout << "num_dims: " << shape->num_dims << std::endl;
    std::cout << "dims: ";
    for (int i = 0; i < shape->num_dims; i++) {
        std::cout << shape->dims[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "strides: ";
    for (int i = 0; i < shape->num_dims; i++) {
        std::cout << shape->strides[i] << " ";
    }
    std::cout << std::endl;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> get_conv_info(ConvInfo* info) {
    std::vector<int64_t> padding(info->padding, info->padding + info->num_dims);
    std::vector<int64_t> stride(info->stride, info->stride + info->num_dims);
    std::vector<int64_t> dilation(info->dilation, info->dilation + info->num_dims);
    return std::make_tuple(padding, stride, dilation);
}

ConvAttributes::ConvAttributes(CudnnTensorShapeStride* x_shape, 
                               CudnnTensorShapeStride* w_shape, 
                               CudnnTensorShapeStride* y_shape, 
                               fe::graph::Graph& graph, 
                               CudnnFrontendDataType_t type,
                               ConvInfo* info) {
    auto data_type = get_data_type(type);

    X = graph.tensor(get_tensor_attributes(from_shape(x_shape->num_dims, x_shape->dims), 
                                           from_shape(x_shape->num_dims, x_shape->strides), 
                                           type));

    W = graph.tensor(get_tensor_attributes(from_shape(w_shape->num_dims, w_shape->dims),
                                           from_shape(w_shape->num_dims, w_shape->strides), 
                                           type));

    auto [padding, stride, dilation] = get_conv_info(info);

    auto conv_options = fe::graph::Conv_fprop_attributes()
                            .set_padding(padding)
                            .set_stride(stride)
                            .set_dilation(dilation);

    Y = graph.conv_fprop(X, W, conv_options);
    Y->set_output(true)
      .set_dim(from_shape(y_shape->num_dims, y_shape->dims))
      .set_stride(from_shape(y_shape->num_dims, y_shape->strides));
}

ConvDescriptor::ConvDescriptor(CudnnFrontendDataType_t type,
                     CudnnTensorShapeStride* x_shape, 
                     CudnnTensorShapeStride* w_shape, 
                     CudnnTensorShapeStride* y_shape, 
                     ConvInfo* info) {
    fe::graph::Graph graph;
    graph.set_io_data_type(get_data_type(type))
         .set_compute_data_type(get_data_type(type));

    attributes = ConvAttributes(x_shape, w_shape, y_shape, graph, type, info);
    this->graph = graph;
}

CudnnFrontendError_t ConvDescriptor::execute(cudnnHandle_t* handle, 
                                             ConvBufers* buffers, 
                                             void* workspace) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {attributes.X, buffers->X},
        {attributes.W, buffers->filter},
        {attributes.Y, buffers->Y}
    };

    return execute_graph(handle, variant_pack, workspace);
}

ConvBkwdDataAttributes::ConvBkwdDataAttributes(CudnnTensorShapeStride* dy_shape, 
                                               CudnnTensorShapeStride* w_shape, 
                                               CudnnTensorShapeStride* dx_shape, 
                                               fe::graph::Graph& graph, 
                                               CudnnFrontendDataType_t type,
                                               ConvInfo* info) {
    auto data_type = get_data_type(type);

    DY = graph.tensor(get_tensor_attributes(from_shape(dy_shape->num_dims, dy_shape->dims), 
                                            from_shape(dy_shape->num_dims, dy_shape->strides), 
                                            type));

    W = graph.tensor(get_tensor_attributes(from_shape(w_shape->num_dims, w_shape->dims),
                                           from_shape(w_shape->num_dims, w_shape->strides), 
                                           type));

    auto [padding, stride, dilation] = get_conv_info(info);

    auto conv_options = fe::graph::Conv_dgrad_attributes()
                            .set_padding(padding)
                            .set_stride(stride)
                            .set_dilation(dilation);

    DX = graph.conv_dgrad(DY, W, conv_options);
    DX->set_output(true)
        .set_dim(from_shape(dx_shape->num_dims, dx_shape->dims))
        .set_stride(from_shape(dx_shape->num_dims, dx_shape->strides));
}

ConvBkwdDataDescriptor::ConvBkwdDataDescriptor(CudnnFrontendDataType_t type,
                                               CudnnTensorShapeStride* dy_shape, 
                                               CudnnTensorShapeStride* w_shape, 
                                               CudnnTensorShapeStride* dx_shape, 
                                               ConvInfo* info) {
    fe::graph::Graph graph;
    graph.set_io_data_type(get_data_type(type))
         .set_compute_data_type(get_data_type(type));

    attributes = ConvBkwdDataAttributes(dy_shape, w_shape, dx_shape, graph, type, info);
    this->graph = graph;
}

CudnnFrontendError_t ConvBkwdDataDescriptor::execute(cudnnHandle_t* handle, ConvBkwdDataBuffers* buffers, void* workspace) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {attributes.DY, buffers->DY},
        {attributes.W, buffers->filter},
        {attributes.DX, buffers->DX}
    };

    return execute_graph(handle, variant_pack, workspace);
}

ConvBkwdFilterAttributes::ConvBkwdFilterAttributes(CudnnTensorShapeStride* x_shape, 
                                                   CudnnTensorShapeStride* dy_shape, 
                                                   CudnnTensorShapeStride* dw_shape, 
                                                   fe::graph::Graph& graph, 
                                                   CudnnFrontendDataType_t type,
                                                   ConvInfo* info) {
    auto data_type = get_data_type(type);

    X = graph.tensor(get_tensor_attributes(from_shape(x_shape->num_dims, x_shape->dims), 
                                           from_shape(x_shape->num_dims, x_shape->strides), 
                                           type));

    DY = graph.tensor(get_tensor_attributes(from_shape(dy_shape->num_dims, dy_shape->dims),
                                            from_shape(dy_shape->num_dims, dy_shape->strides), 
                                            type));

    auto [padding, stride, dilation] = get_conv_info(info);

    auto conv_options = fe::graph::Conv_wgrad_attributes()
                            .set_padding(padding)
                            .set_stride(stride)
                            .set_dilation(dilation);

    DW = graph.conv_wgrad(DY, X, conv_options);
    DW->set_output(true)
        .set_dim(from_shape(dw_shape->num_dims, dw_shape->dims))
        .set_stride(from_shape(dw_shape->num_dims, dw_shape->strides));
}

ConvBkwdFilterDescriptor::ConvBkwdFilterDescriptor(CudnnFrontendDataType_t type,
                                                   CudnnTensorShapeStride* x_shape, 
                                                   CudnnTensorShapeStride* dy_shape, 
                                                   CudnnTensorShapeStride* dw_shape, 
                                                   ConvInfo* info) {
    fe::graph::Graph graph;
    graph.set_io_data_type(get_data_type(type))
         .set_compute_data_type(get_data_type(type));

    attributes = ConvBkwdFilterAttributes(x_shape, dy_shape, dw_shape, graph, type, info);
    this->graph = graph;
}

CudnnFrontendError_t ConvBkwdFilterDescriptor::execute(cudnnHandle_t* handle, 
                                                       ConvBkwdFilterBuffers* buffers, 
                                                       void* workspace) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {attributes.X, buffers->X},
        {attributes.DY, buffers->DY},
        {attributes.DW, buffers->DW}
    };

    return execute_graph(handle, variant_pack, workspace);
}

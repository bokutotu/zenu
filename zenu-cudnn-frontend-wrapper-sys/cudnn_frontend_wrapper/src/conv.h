#pragma once

#include "../include/cudnn_frontend_wrapper.h"
#include  "i_graph_desc.h"
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

struct ConvAttributes {
    std::shared_ptr<fe::graph::Tensor_attributes> X;
    std::shared_ptr<fe::graph::Tensor_attributes> W;
    std::shared_ptr<fe::graph::Tensor_attributes> Y;

    ConvAttributes() = default;

    ConvAttributes(CudnnTensorShapeStride* x_shape, 
                   CudnnTensorShapeStride* w_shape, 
                   CudnnTensorShapeStride* y_shape, 
                   fe::graph::Graph& graph, 
                   CudnnFrontendDataType_t type,
                   ConvInfo* info);
};

struct ConvDescriptor : public IGraphDescriptor {
private:
    ConvAttributes attributes;

public:
    ConvDescriptor(CudnnFrontendDataType_t type, 
              CudnnTensorShapeStride* x_shape, 
              CudnnTensorShapeStride* w_shape, 
              CudnnTensorShapeStride* y_shape, 
              ConvInfo* info);

    CudnnFrontendError_t execute(cudnnHandle_t* handle, 
                                 ConvBufers* buffers, 
                                 void* workspace);
};

struct ConvBkwdDataAttributes {
    std::shared_ptr<fe::graph::Tensor_attributes> DY;
    std::shared_ptr<fe::graph::Tensor_attributes> W;
    std::shared_ptr<fe::graph::Tensor_attributes> DX;

    ConvBkwdDataAttributes() = default;

    ConvBkwdDataAttributes(CudnnTensorShapeStride* dy_shape, 
                           CudnnTensorShapeStride* w_shape, 
                           CudnnTensorShapeStride* dx_shape, 
                           fe::graph::Graph& graph, 
                           CudnnFrontendDataType_t type,
                           ConvInfo* info);
};

struct ConvBkwdDataDescriptor : public IGraphDescriptor {
private:
    ConvBkwdDataAttributes attributes;

public:
    ConvBkwdDataDescriptor(CudnnFrontendDataType_t type, 
                           CudnnTensorShapeStride* dy_shape, 
                           CudnnTensorShapeStride* w_shape, 
                           CudnnTensorShapeStride* dx_shape, 
                           ConvInfo* info);

    CudnnFrontendError_t execute(cudnnHandle_t* handle, 
                                 ConvBkwdDataBuffers* buffers, 
                                 void* workspace);
};

struct ConvBkwdFilterAttributes {
    std::shared_ptr<fe::graph::Tensor_attributes> X;
    std::shared_ptr<fe::graph::Tensor_attributes> DY;
    std::shared_ptr<fe::graph::Tensor_attributes> DW;

    ConvBkwdFilterAttributes() = default;

    ConvBkwdFilterAttributes(CudnnTensorShapeStride* x_shape, 
                             CudnnTensorShapeStride* dy_shape, 
                             CudnnTensorShapeStride* dw_shape, 
                             fe::graph::Graph& graph, 
                             CudnnFrontendDataType_t type,
                             ConvInfo* info);
};

struct ConvBkwdFilterDescriptor : public IGraphDescriptor {
private:
    ConvBkwdFilterAttributes attributes;

public:
    ConvBkwdFilterDescriptor(CudnnFrontendDataType_t type, 
                             CudnnTensorShapeStride* x_shape, 
                             CudnnTensorShapeStride* dy_shape, 
                             CudnnTensorShapeStride* dw_shape, 
                             ConvInfo* info);

    CudnnFrontendError_t execute(cudnnHandle_t* handle, 
                                 ConvBkwdFilterBuffers* buffers, 
                                 void* workspace);
};

#include "../include/cudnn_frontend_wrapper.h"

#include "batchnorm.h"
#include "conv.h"

CudnnFrontendError_t create_batch_norm_descriptor(BatchNormDescriptor** desc, 
                                                  CudnnFrontendDataType_t data_type, 
                                                  const CudnnTensorShapeStride* shape,
                                                  float epsilon,
                                                  float momentum,
                                                  bool is_training) {

    *desc = new BatchNormDescriptor(*shape, is_training, data_type, epsilon, momentum);
    return CudnnFrontendError_t::SUCCESS;
}

CudnnFrontendError_t get_workspace_size(BatchNormDescriptor* desc, int64_t* workspace_size) {
    return desc->get_workspace_size(workspace_size);
}

CudnnFrontendError_t check_graph(BatchNormDescriptor* desc, cudnnHandle_t* handle) {
    return desc->check_graph(handle);
}

void batch_norm_desc_debug(BatchNormDescriptor* desc) {
    desc->debug_print();
}

CudnnFrontendError_t execute_batch_norm_forward_training(BatchNormDescriptor* desc, 
                                                         BatchNormExecutionBuffers* buffers,
                                                         void* workspace,
                                                         cudnnHandle_t* handle) {
    return desc->execute(handle, buffers, workspace);
}

CudnnFrontendError_t create_batch_norm_backward_data_descriptor(BatchNormBkwdDescriptor** desc,
                                                               CudnnFrontendDataType_t data_type,
                                                               const CudnnTensorShapeStride* shape) {
    *desc = new BatchNormBkwdDescriptor(*shape, data_type);
    return CudnnFrontendError_t::SUCCESS;
}

CudnnFrontendError_t get_backward_data_workspace_size(BatchNormBkwdDescriptor* desc, 
                                                      int64_t* workspace_size) {
    return desc->get_workspace_size(workspace_size);
}

CudnnFrontendError_t check_backward_data_graph(BatchNormBkwdDescriptor* desc, 
                                               cudnnHandle_t* handle) {
    return desc->check_graph(handle);
}

CudnnFrontendError_t execute_batch_norm_backward_data(BatchNormBkwdDescriptor* desc,
                                                      BatchNormBkwdExecutionBuffers* buffers,
                                                      void* workspace,
                                                      cudnnHandle_t* handle) {
    return desc->execute(handle, buffers, workspace);
}

CudnnFrontendError_t create_conv_descriptor(ConvDescriptor** desc, 
                                            CudnnFrontendDataType_t data_type, 
                                            CudnnTensorShapeStride* x_shape,
                                            CudnnTensorShapeStride* w_shape,
                                            CudnnTensorShapeStride* y_shape,
                                            ConvInfo* info) {
    *desc = new ConvDescriptor(data_type, x_shape, w_shape, y_shape, info);
    return CudnnFrontendError_t::SUCCESS;
}

CudnnFrontendError_t get_conv_workspace_size(ConvDescriptor* desc, int64_t* workspace_size) {
    return desc->get_workspace_size(workspace_size);
}

CudnnFrontendError_t check_conv_graph(ConvDescriptor* desc, cudnnHandle_t* handle) {
    return desc->build_and_check_graph(handle, false);
}

CudnnFrontendError_t execute_conv_forward(ConvDescriptor* desc, 
                                          ConvBufers* buffers,
                                          void* workspace,
                                          cudnnHandle_t* handle) {
    return desc->execute(handle, buffers, workspace);
}

CudnnFrontendError_t create_conv_backward_data_descriptor(ConvBkwdDataDescriptor** desc, 
                                                          CudnnFrontendDataType_t data_type, 
                                                          CudnnTensorShapeStride* dy_shape,
                                                          CudnnTensorShapeStride* w_shape,
                                                          CudnnTensorShapeStride* dx_shape,
                                                          ConvInfo* info) {
    *desc = new ConvBkwdDataDescriptor(data_type, dy_shape, w_shape, dx_shape, info);
    return CudnnFrontendError_t::SUCCESS;
}

CudnnFrontendError_t get_conv_backward_data_workspace_size(ConvBkwdDataDescriptor* desc, 
                                                           int64_t* workspace_size) {
    return desc->get_workspace_size(workspace_size);
}

CudnnFrontendError_t check_conv_backward_data_graph(ConvBkwdDataDescriptor* desc,
                                                    cudnnHandle_t* handle) {
    return desc->build_and_check_graph(handle, false);
}

CudnnFrontendError_t execute_conv_backward_data(ConvBkwdDataDescriptor* desc,
                                                ConvBkwdDataBuffers* buffers,
                                                void* workspace,
                                                cudnnHandle_t* handle) {
    return desc->execute(handle, buffers, workspace);
}

CudnnFrontendError_t create_conv_backward_filter_descriptor(ConvBkwdFilterDescriptor** desc, 
                                                            CudnnFrontendDataType_t data_type, 
                                                            CudnnTensorShapeStride* x_shape,
                                                            CudnnTensorShapeStride* dy_shape,
                                                            CudnnTensorShapeStride* dw_shape,
                                                            ConvInfo* info) {
    *desc = new ConvBkwdFilterDescriptor(data_type, x_shape, dy_shape, dw_shape, info);
    return CudnnFrontendError_t::SUCCESS;
}

CudnnFrontendError_t get_conv_backward_filter_workspace_size(ConvBkwdFilterDescriptor* desc, 
                                                             int64_t* workspace_size) {
    return desc->get_workspace_size(workspace_size);
}

CudnnFrontendError_t check_conv_backward_filter_graph(ConvBkwdFilterDescriptor* desc,
                                                      cudnnHandle_t* handle) {
    return desc->build_and_check_graph(handle, false);
}

CudnnFrontendError_t execute_conv_backward_filter(ConvBkwdFilterDescriptor* desc,
                                                  ConvBkwdFilterBuffers* buffers,
                                                  void* workspace,
                                                  cudnnHandle_t* handle) {
    return desc->execute(handle, buffers, workspace);
}

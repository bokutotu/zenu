#include <catch2/catch.hpp>

#include "cudnn_frontend_wrapper.h"
#include <cudnn.h>

#include <cstdlib>
#include "helpers.h"

TEST_CASE("conv2d", "[conv2d]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

    CudnnTensorShapeStride shape = {
        .num_dims = 4,
        .dims = {n, c, h, w},
        .strides = {c * h * w, h * w, w, 1}
    };
    CudnnTensorShapeStride filter_shape = {
        .num_dims = 4,
        .dims = {k, c, r, s},
        .strides = {c * r * s, r * s, s, 1}
    };
    CudnnTensorShapeStride y_shape = {
        .num_dims = 4,
        .dims = {n, k, h, w},
        .strides = {k * h * w, h * w, w, 1}
    };
    ConvInfo info = {
        .padding = {0, 0},
        .stride = {1, 1},
        .dilation = {1, 1},
        .num_dims = 2
    };
    ConvDescriptor* desc;
    CudnnFrontendError_t status = create_conv_descriptor(&desc, 
                                                         DATA_TYPE_FLOAT, 
                                                         &shape, 
                                                         &filter_shape, 
                                                         &y_shape, 
                                                         &info);
    REQUIRE(status == SUCCESS);
    CHECK(desc != nullptr);

    status = check_conv_graph(desc, &handle);
    REQUIRE(status == SUCCESS);

    int64_t workspace_size;
    status = get_conv_workspace_size(desc, &workspace_size);
    REQUIRE(status == SUCCESS);

    ConvBufers buffers = {
        .X = nullptr,
        .filter = nullptr,
        .Y = nullptr
    };

    Surface<float> X_tensor(n * c * h * w, false);
    Surface<float> Filter_tensor(k * c * r * s, false);
    Surface<float> Y_tensor(n * k * h * w, false);
    Surface<int8_t> workspace(workspace_size, false);

    buffers.X = X_tensor.devPtr;
    buffers.filter = Filter_tensor.devPtr;
    buffers.Y = Y_tensor.devPtr;

    status = execute_conv_forward(desc, &buffers, workspace.devPtr, &handle);
    REQUIRE(status == SUCCESS);
};

TEST_CASE("conv2d backward data", "[conv2d_backwawrd_data]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

    CudnnTensorShapeStride dy_shape = {
        .num_dims = 4,
        .dims = {n, k, h, w},
        .strides = {k * h * w, h * w, w, 1}
    };
    CudnnTensorShapeStride filter_shape = {
        .num_dims = 4,
        .dims = {k, c, r, s},
        .strides = {c * r * s, r * s, s, 1}
    };
    CudnnTensorShapeStride dx_shape = {
        .num_dims = 4,
        .dims = {n, c, h, w},
        .strides = {c * h * w, h * w, w, 1}
    };
    ConvInfo info = {
        .padding = {0, 0},
        .stride = {1, 1},
        .dilation = {1, 1},
        .num_dims = 2
    };
    ConvBkwdDataDescriptor* desc;
    CudnnFrontendError_t status = create_conv_backward_data_descriptor(&desc, 
                                                                      DATA_TYPE_FLOAT, 
                                                                      &dy_shape, 
                                                                      &filter_shape, 
                                                                      &dx_shape, 
                                                                      &info);
    REQUIRE(status == SUCCESS);
    CHECK(desc != nullptr);

    status = check_conv_backward_data_graph(desc, &handle);
    REQUIRE(status == SUCCESS);

    int64_t workspace_size;
    status = get_conv_backward_data_workspace_size(desc, &workspace_size);
    REQUIRE(status == SUCCESS);

    ConvBkwdDataBuffers buffers = {
        .DY = nullptr,
        .filter = nullptr,
        .DX = nullptr
    };

    Surface<float> DY_tensor(n * k * h * w, false);
    Surface<float> Filter_tensor(k * c * r * s, false);
    Surface<float> DX_tensor(n * c * h * w, false);
    Surface<int8_t> workspace(workspace_size, false);

    buffers.DY = DY_tensor.devPtr;
    buffers.filter = Filter_tensor.devPtr;
    buffers.DX = DX_tensor.devPtr;

    status = execute_conv_backward_data(desc, &buffers, workspace.devPtr, &handle);
    REQUIRE(status == SUCCESS);
};

TEST_CASE("conv2d backward fileter", "[conv2d_backward_fileter]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 3, s = 3;

    CudnnTensorShapeStride x_shape = {
        .num_dims = 4,
        .dims = {n, c, h, w},
        .strides = {c * h * w, h * w, w, 1}
    };
    CudnnTensorShapeStride dy_shape = {
        .num_dims = 4,
        .dims = {n, k, h, w},
        .strides = {k * h * w, h * w, w, 1}
    };
    CudnnTensorShapeStride dw_shape = {
        .num_dims = 4,
        .dims = {k, c, r, s},
        .strides = {c * r * s, r * s, s, 1}
    };
    ConvInfo info = {
        .padding = {1, 1},
        .stride = {1, 1},
        .dilation = {1, 1},
        .num_dims = 2
    };
    ConvBkwdFilterDescriptor* desc;
    CudnnFrontendError_t status = create_conv_backward_filter_descriptor(&desc, 
                                                                        DATA_TYPE_FLOAT, 
                                                                        &x_shape, 
                                                                        &dy_shape, 
                                                                        &dw_shape, 
                                                                        &info);
    REQUIRE(status == SUCCESS);
    CHECK(desc != nullptr);

    status = check_conv_backward_filter_graph(desc, &handle);
    REQUIRE(status == SUCCESS);

    int64_t workspace_size;
    status = get_conv_backward_filter_workspace_size(desc, &workspace_size);
    REQUIRE(status == SUCCESS);

    ConvBkwdFilterBuffers buffers = {
        .X = nullptr,
        .DY = nullptr,
        .DW = nullptr
    };

    Surface<float> X_tensor(n * c * h * w, false);
    Surface<float> DY_tensor(n * k * h * w, false);
    Surface<float> DW_tensor(k * c * r * s, false);
    Surface<int8_t> workspace(workspace_size, false);

    buffers.X = X_tensor.devPtr;
    buffers.DY = DY_tensor.devPtr;
    buffers.DW = DW_tensor.devPtr;

    status = execute_conv_backward_filter(desc, &buffers, workspace.devPtr, &handle);
    REQUIRE(status == SUCCESS);
    cudnnDestroy(handle);
};


TEST_CASE("conv1d forward", "[conv1d_forward]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    int64_t n = 16, c = 128, w = 64, k = 256, r = 3; 
    CudnnTensorShapeStride x_shape = {
        .num_dims = 3,
        .dims = {n, c, w},
        .strides = {c*w, w, 1}
    };
    CudnnTensorShapeStride w_shape = {
        .num_dims = 3,
        .dims = {k, c, r},
        .strides = {c*r, r, 1}
    };
    ConvInfo info = {
        .padding = {1},   
        .stride = {1},
        .dilation = {1},
        .num_dims = 1
    };
    CudnnTensorShapeStride y_shape = {
        .num_dims = 3,
        .dims = {n, k, w},
        .strides = {k*w, w, 1}
    };

    ConvDescriptor* desc;
    CudnnFrontendError_t status = create_conv_descriptor(&desc,
                                                         DATA_TYPE_FLOAT,
                                                         &x_shape,
                                                         &w_shape,
                                                         &y_shape,
                                                         &info);
    REQUIRE(status == SUCCESS);
    CHECK(desc != nullptr);

    status = check_conv_graph(desc, &handle);
    REQUIRE(status == SUCCESS);

    int64_t workspace_size;
    status = get_conv_workspace_size(desc, &workspace_size);
    REQUIRE(status == SUCCESS);

    ConvBufers buffers = {
        .X = nullptr,
        .filter = nullptr,
        .Y = nullptr
    };

    Surface<float> X_tensor(n * c * w, false);
    Surface<float> Filter_tensor(k * c * r, false);
    Surface<float> Y_tensor(n * k * w, false);
    Surface<int8_t> workspace(workspace_size, false);

    buffers.X = X_tensor.devPtr;
    buffers.filter = Filter_tensor.devPtr;
    buffers.Y = Y_tensor.devPtr;

    status = execute_conv_forward(desc, &buffers, workspace.devPtr, &handle);
    REQUIRE(status == SUCCESS);

    cudnnDestroy(handle);
}

TEST_CASE("conv1d backward data", "[conv1d_backward_data]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    int64_t n = 16, c = 128, w = 64, k = 256, r = 3;
    ConvInfo info = {
        .padding = {1},   
        .stride = {1},
        .dilation = {1},
        .num_dims = 1
    };

    CudnnTensorShapeStride dy_shape = {
        .num_dims = 3,
        .dims = {n, k, w},
        .strides = {k*w, w, 1}
    };
    CudnnTensorShapeStride filter_shape = {
        .num_dims = 3,
        .dims = {k, c, r},
        .strides = {c*r, r, 1}
    };
    CudnnTensorShapeStride dx_shape = {
        .num_dims = 3,
        .dims = {n, c, w},
        .strides = {c*w, w, 1}
    };

    ConvBkwdDataDescriptor* desc;
    CudnnFrontendError_t status = create_conv_backward_data_descriptor(&desc,
                                                                      DATA_TYPE_FLOAT,
                                                                      &dy_shape,
                                                                      &filter_shape,
                                                                      &dx_shape,
                                                                      &info);
    REQUIRE(status == SUCCESS);
    CHECK(desc != nullptr);

    status = check_conv_backward_data_graph(desc, &handle);
    REQUIRE(status == SUCCESS);

    int64_t workspace_size;
    status = get_conv_backward_data_workspace_size(desc, &workspace_size);
    REQUIRE(status == SUCCESS);

    ConvBkwdDataBuffers buffers = {
        .DY = nullptr,
        .filter = nullptr,
        .DX = nullptr
    };

    Surface<float> DY_tensor(n * k * w, false);
    Surface<float> Filter_tensor(k * c * r, false);
    Surface<float> DX_tensor(n * c * w, false);
    Surface<int8_t> workspace(workspace_size, false);

    buffers.DY = DY_tensor.devPtr;
    buffers.filter = Filter_tensor.devPtr;
    buffers.DX = DX_tensor.devPtr;

    status = execute_conv_backward_data(desc, &buffers, workspace.devPtr, &handle);
    REQUIRE(status == SUCCESS);

    cudnnDestroy(handle);
}

TEST_CASE("conv1d backward filter", "[conv1d_backward_filter]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    int64_t n = 16, c = 128, w = 64, k = 256, r = 3;
    ConvInfo info = {
        .padding = {1},   
        .stride = {1},
        .dilation = {1},
        .num_dims = 1
    };

    CudnnTensorShapeStride x_shape = {
        .num_dims = 3,
        .dims = {n, c, w},
        .strides = {c*w, w, 1}
    };
    CudnnTensorShapeStride dy_shape = {
        .num_dims = 3,
        .dims = {n, k, w},
        .strides = {k*w, w, 1}
    };
    CudnnTensorShapeStride dw_shape = {
        .num_dims = 3,
        .dims = {k, c, r},
        .strides = {c*r, r, 1}
    };

    ConvBkwdFilterDescriptor* desc;
    CudnnFrontendError_t status = create_conv_backward_filter_descriptor(&desc,
                                                                        DATA_TYPE_FLOAT,
                                                                        &x_shape,
                                                                        &dy_shape,
                                                                        &dw_shape,
                                                                        &info);
    REQUIRE(status == SUCCESS);
    CHECK(desc != nullptr);

    status = check_conv_backward_filter_graph(desc, &handle);
    REQUIRE(status == SUCCESS);

    int64_t workspace_size;
    status = get_conv_backward_filter_workspace_size(desc, &workspace_size);
    REQUIRE(status == SUCCESS);

    ConvBkwdFilterBuffers buffers = {
        .X = nullptr,
        .DY = nullptr,
        .DW = nullptr
    };

    Surface<float> X_tensor(n * c * w, false);
    Surface<float> DY_tensor(n * k * w, false);
    Surface<float> DW_tensor(k * c * r, false);
    Surface<int8_t> workspace(workspace_size, false);

    buffers.X = X_tensor.devPtr;
    buffers.DY = DY_tensor.devPtr;
    buffers.DW = DW_tensor.devPtr;

    status = execute_conv_backward_filter(desc, &buffers, workspace.devPtr, &handle);
    REQUIRE(status == SUCCESS);

    cudnnDestroy(handle);
};

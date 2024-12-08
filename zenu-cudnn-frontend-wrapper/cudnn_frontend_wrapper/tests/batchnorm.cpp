#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "cudnn_frontend_wrapper.h"
#include <cudnn.h>

#include <cstdlib>
#include "helpers.h"

TEST_CASE("BatchNorm2d", "[batchnorm2d]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    CudnnTensorShapeStride shape = {
        .num_dims = 4,
        .dims = {4, 32, 16, 16},
        .strides = {32*16*16, 16*16, 16, 1}
    };
    float epsilon = 1e-5;
    float momentum = 0.1;
    bool is_training = true;
    BatchNormDescriptor* desc;
    CudnnFrontendError_t status = create_batch_norm_descriptor(&desc, 
                                                               DATA_TYPE_FLOAT, 
                                                               &shape, 
                                                               epsilon, 
                                                               momentum, 
                                                               is_training);
    REQUIRE(status == SUCCESS);
    CHECK(desc != nullptr);
    status = check_graph(desc, &handle);
    REQUIRE(status == SUCCESS);
    int64_t workspace_size;
    status = get_workspace_size(desc, &workspace_size);
    REQUIRE(status == SUCCESS);
    CHECK(workspace_size > 0);

    BatchNormExecutionBuffers buffers = {
        .X = nullptr,
        .mean = nullptr,
        .inv_variance = nullptr,
        .scale = nullptr,
        .bias = nullptr,
        .peer_stats_0 = nullptr,
        .peer_stats_1 = nullptr,
        .prev_running_mean = nullptr,
        .prev_running_var = nullptr,
        .next_running_mean = nullptr,
        .next_running_var = nullptr,
        .Y = nullptr
    };

    Surface<float> X_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Var_tensor(32, false);
    Surface<float> Previous_running_mean_tensor(32, false);
    Surface<float> Previous_running_var_tensor(32, false);
    Surface<float> Next_running_mean_tensor(32, false);
    Surface<float> Next_running_var_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Bias_tensor(32, false);
    Surface<float> Y_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Peer_stats_0_tensor(2 * 4 * 32, false, true);
    Surface<float> Peer_stats_1_tensor(2 * 4 * 32, false);

    buffers.X = X_tensor.devPtr;
    buffers.mean = Mean_tensor.devPtr;
    buffers.inv_variance = Var_tensor.devPtr;
    buffers.scale = Scale_tensor.devPtr;
    buffers.bias = Bias_tensor.devPtr;
    buffers.peer_stats_0 = Peer_stats_0_tensor.devPtr;
    buffers.peer_stats_1 = Peer_stats_1_tensor.devPtr;
    buffers.prev_running_mean = Previous_running_mean_tensor.devPtr;
    buffers.prev_running_var = Previous_running_var_tensor.devPtr;
    buffers.next_running_mean = Next_running_mean_tensor.devPtr;
    buffers.next_running_var = Next_running_var_tensor.devPtr;
    buffers.Y = Y_tensor.devPtr;

    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);
    cudaDeviceSynchronize();
    status = execute_batch_norm_forward_training(desc, &buffers, workspace, &handle);
    REQUIRE(status == SUCCESS);
};

TEST_CASE("BatchNorm2dBkwd", "[batchnorm2d_bkwd]") {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    CudnnTensorShapeStride shape = {
        .num_dims = 4,
        .dims = {4, 32, 16, 16},
        .strides = {32*16*16, 16*16, 16, 1}
    };

    BatchNormBkwdDescriptor* desc;
    CudnnFrontendError_t status = create_batch_norm_backward_data_descriptor(&desc, 
                                                                           DATA_TYPE_FLOAT, 
                                                                           &shape);
    REQUIRE(status == SUCCESS);
    CHECK(desc != nullptr);
    status = check_backward_data_graph(desc, &handle);
    REQUIRE(status == SUCCESS);
    int64_t workspace_size;
    status = get_backward_data_workspace_size(desc, &workspace_size);
    REQUIRE(status == SUCCESS);
    CHECK(workspace_size > 0);

    BatchNormBkwdExecutionBuffers buffers = {
        .X = nullptr,
        .DY = nullptr,
        .scale = nullptr,
        .mean = nullptr,
        .inv_variance = nullptr,
        .dscale = nullptr,
        .dbias = nullptr,
        .DX = nullptr,
        .peer_stats_0 = nullptr,
        .peer_stats_1 = nullptr
    };

    Surface<float> X_tensor(4 * 32 * 16 * 16, false);
    Surface<float> DY_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Inv_variance_tensor(32, false);
    Surface<float> Dscale_tensor(32, false);
    Surface<float> Dbias_tensor(32, false);
    Surface<float> DX_tensor(4 * 32 * 16 * 16, false);
    Surface<float> Peer_stats_0_tensor(2 * 4 * 32, false, true);
    Surface<float> Peer_stats_1_tensor(2 * 4 * 32, false);

    buffers.X = X_tensor.devPtr;
    buffers.DY = DY_tensor.devPtr;
    buffers.scale = Scale_tensor.devPtr;
    buffers.mean = Mean_tensor.devPtr;
    buffers.inv_variance = Inv_variance_tensor.devPtr;
    buffers.dscale = Dscale_tensor.devPtr;
    buffers.dbias = Dbias_tensor.devPtr;
    buffers.DX = DX_tensor.devPtr;
    buffers.peer_stats_0 = Peer_stats_0_tensor.devPtr;
    buffers.peer_stats_1 = Peer_stats_1_tensor.devPtr;

    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);
    cudaDeviceSynchronize();
    status = execute_batch_norm_backward_data(desc, &buffers, workspace, &handle);
    REQUIRE(status == SUCCESS);
}

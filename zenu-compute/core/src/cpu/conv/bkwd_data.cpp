#include "conv.h"

#include "zenu_compute_blas.h"

#include <cstddef>
#include <stdint.h>
#include <cstring>
#include <iostream>

std::array<size_t, 3> ZenuComputeConvCpuImpl::get_gemm_param_bkwd_data() const {
    if (get_dim() == 2) {
        return get_gemm_param2d_bkwd_data();
    } else {
        return {0, 0, 0};
    }
}

std::array<size_t, 3> ZenuComputeConvCpuImpl::get_gemm_param2d_bkwd_data() const {
    // [M, K, N] for backward data GEMM
    // M = C*R*S (input channels × kernel spatial size)
    // K = K (output channels)
    // N = N*P*Q (batch × output spatial size)
    return {
        input[1] * kernel[2] * kernel[3], // C*R*S
        kernel[0],                         // K
        input[0] * output[2] * output[3]   // N*P*Q
    };
}

ZenuStatus ZenuComputeConvCpuImpl::backward_data(const void* kernel, const void* grad_output, void* grad_input, void* workspace) const {
    // Null check
    if (!kernel || !grad_output || !grad_input || !workspace) {
        return InvalidArgument;
    }

    memset(workspace, 0, get_backward_data_bytes());
    size_t input_size = 1;
    for (size_t i = 0; i < input.size(); i++) {
        input_size *= input[i];
    }
    memset(grad_input, 0, input_size * ((type == f32) ? 4 : 8));

    const size_t dtype_size = (type == f32) ? 4 : 8;
    const auto [M, K, N] = get_gemm_param_bkwd_data();

    transpose_gemm2d_bkwd_data(grad_output, workspace);

    size_t grad_output_num_elm = this->output[0] * this->output[1] * this->output[2] * this->output[3];
    char* gemm_out = static_cast<char*>(workspace) + grad_output_num_elm * dtype_size + 1024;
    gemm_out -= reinterpret_cast<size_t>(gemm_out) % 64;

    zenu_compute_gemm_cpu(
        Transpose,
        NoTranspose,
        M, N, K,
        1.0f,
        kernel, 
        M,
        workspace,
        N,
        0.0f,
        gemm_out,
        N,
        type
    );

    col2im2d(gemm_out, grad_input);

    return Success;
}

void ZenuComputeConvCpuImpl::transpose_gemm_bkwd_data(const void* d_output, void* d_output_reshaped) const {
    if (get_dim() == 2) {
        transpose_gemm2d_bkwd_data(d_output, d_output_reshaped);
    }
}

void ZenuComputeConvCpuImpl::transpose_gemm2d_bkwd_data(const void* d_output, void* d_output_reshaped) const {
    const size_t N = input[0];
    const size_t K = kernel[0];
    const size_t PQ = output[2] * output[3];
    const size_t elem_size = (type == f32) ? 4 : 8;

    // Transpose [N][K][PQ] -> [K][N][PQ]
    #pragma omp parallel for collapse(2)
    for (size_t k = 0; k < K; k++) {
        for (size_t n = 0; n < N; n++) {
            const size_t src_offset = (n * K + k) * PQ * elem_size;
            const size_t dst_offset = (k * N + n) * PQ * elem_size;
            
            memcpy(
                static_cast<uint8_t*>(d_output_reshaped) + dst_offset,
                static_cast<const uint8_t*>(d_output) + src_offset,
                PQ * elem_size
            );
        }
    }
}
size_t ZenuComputeConvCpuImpl::get_backward_data_bytes() const {
    size_t num_elm_output = 1;
    for (int i = 0; i < output.size(); i++) {
        num_elm_output *= output[i];
    }
    size_t data_bytes;
    switch (type) {
    case ZenuDataType::f32:
        data_bytes = sizeof(float);
        break;
    case ZenuDataType::f64:
        data_bytes = sizeof(double);
        break;
    default:
        std::cout << "Unsupported data type" << std::endl;
        exit(1);
    }
    return get_gemm_bytes_bkwd_data() + get_im2col_bytes() + 1024;
}

size_t ZenuComputeConvCpuImpl::get_gemm_bytes_bkwd_data() const {
    const auto [M, K, N] = get_gemm_param_bkwd_data();
    auto gemm_output_num_elm = M * N;
    size_t data_bytes;
    switch (type) {
    case ZenuDataType::f32:
        data_bytes = sizeof(float);
        break;
    case ZenuDataType::f64:
        data_bytes = sizeof(double);
        break;
    default:
        std::cout << "Unsupported data type" << std::endl;
        exit(1);
    }
    return gemm_output_num_elm * data_bytes;
}

#include "conv.h"
#include "zenu_compute_blas.h"

#include <cstddef>
#include <stdint.h>
#include <cstring>
#include <iostream>
#include <omp.h>

static void transpose_2d_array(
    const void* src,
    void* dst,
    size_t rows,
    size_t cols,
    size_t elem_size
) {
#pragma omp parallel for collapse(2)
    for(size_t r = 0; r < rows; r++){
        for(size_t c = 0; c < cols; c++){
            const size_t src_idx = r * cols + c;
            const size_t dst_idx = c * rows + r;
            // 1要素分を memcpy
            memcpy(
                static_cast<uint8_t*>(dst) + dst_idx * elem_size,
                static_cast<const uint8_t*>(src) + src_idx * elem_size,
                elem_size
            );
        }
    }
}


std::array<size_t, 3> ZenuComputeConvCpuImpl::get_gemm_param_bkwd_data() const {
    if (get_dim() == 2) {
        return get_gemm_param2d_bkwd_data();
    } else {
        return {0, 0, 0};
    }
}

std::array<size_t, 3> ZenuComputeConvCpuImpl::get_gemm_param2d_bkwd_data() const {
    return {
        input[1] * kernel[2] * kernel[3],
        kernel[0],
        input[0] * output[2] * output[3]
    };
}

ZenuStatus ZenuComputeConvCpuImpl::backward_data(
    const void* kernel,
    const void* grad_output,
    void* grad_input,
    void* workspace
) const
{
    if (!kernel || !grad_output || !grad_input || !workspace) {
        return InvalidArgument;
    }

    memset(workspace, 0, get_backward_data_bytes());

    size_t input_size = 1;
    for (size_t i = 0; i < input.size(); i++) {
        input_size *= input[i];
    }
    memset(grad_input, 0, input_size * ((type == f32) ? 4 : 8));

    const auto [M, K, N] = get_gemm_param_bkwd_data(); 
    const size_t dtype_size = (type == f32) ? 4 : 8;

    transpose_gemm2d_bkwd_data(grad_output, workspace);

    size_t grad_output_num_elm = this->output[0] * this->output[1] * this->output[2] * this->output[3];
    char* gemm_out = static_cast<char*>(workspace) + grad_output_num_elm * dtype_size + 1024;
    gemm_out -= reinterpret_cast<size_t>(gemm_out) % 64;

    zenu_compute_gemm_cpu(
        Transpose,
        NoTranspose,
        M,
        N,
        K,
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

    const size_t c_r_s = M;
    const size_t n_p_q = N;

    char* col_buf = gemm_out + (c_r_s * n_p_q * dtype_size);
    col_buf += 1024;
    col_buf -= reinterpret_cast<size_t>(col_buf) % 64;

    transpose_2d_array(
        gemm_out,
        col_buf,
        c_r_s,
        n_p_q,
        dtype_size
    );

    col2im2d(col_buf, grad_input);

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

    #pragma omp parallel for collapse(2)
    for (size_t k_idx = 0; k_idx < K; k_idx++) {
        for (size_t n_idx = 0; n_idx < N; n_idx++) {
            const size_t src_offset = (n_idx * K + k_idx) * PQ * elem_size;
            const size_t dst_offset = (k_idx * (N * PQ) + n_idx * PQ) * elem_size;
            memcpy(
                static_cast<uint8_t*>(d_output_reshaped) + dst_offset,
                static_cast<const uint8_t*>(d_output)    + src_offset,
                PQ * elem_size
            );
        }
    }
}

size_t ZenuComputeConvCpuImpl::get_backward_data_bytes() const {
    const auto [M, K, N] = get_gemm_param_bkwd_data();
    size_t gemm_out_bytes = M * N;
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

    size_t gemm_buf_total = gemm_out_bytes * data_bytes * 2;

    size_t d_output_size = 1;
    for (auto &dim : output) d_output_size *= dim;
    size_t d_output_bytes = d_output_size * data_bytes;

    size_t total = d_output_bytes + gemm_buf_total + get_im2col_bytes() + 2048; 
    return total;
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


#include "conv.h"
#include "zenu_compute_blas.h"
#include "macro.h"

#include <cstddef>
#include <stdint.h>
#include <cstring>
#include <omp.h>


std::array<size_t, 3> ZenuComputeConvCpuImpl::get_gemm_param_bkwd_kernel() const {
    if (get_dim() == 2) {
        return get_gemm_param2d_bkwd_kernel();
    } else {
        return {0,0,0};
    }
}

std::array<size_t, 3> ZenuComputeConvCpuImpl::get_gemm_param2d_bkwd_kernel() const {
    const size_t M = kernel[0];
    const size_t KK = input[0] * output[2] * output[3];  
    const size_t N = input[1] * kernel[2] * kernel[3];
    return {M, KK, N};
}

void ZenuComputeConvCpuImpl::transpose_gemm2d_bkwd_kernel(
    const void* grad_output, 
    void*       grad_output_reshaped
) const 
{
    const size_t N  = input[0];
    const size_t K  = kernel[0];
    const size_t PQ = output[2] * output[3];

    size_t elem_size;
    DEFINE_DATA_SIZE(type, elem_size);

#pragma omp parallel for collapse(2)
    for (size_t k_idx = 0; k_idx < K; k_idx++) {
        for (size_t n_idx = 0; n_idx < N; n_idx++) {
            const size_t src_offset = (n_idx * K + k_idx) * PQ * elem_size;
            const size_t dst_offset = (k_idx * (N * PQ) + n_idx * PQ) * elem_size;

            memcpy(
                static_cast<uint8_t*>(grad_output_reshaped) + dst_offset,
                static_cast<const uint8_t*>(grad_output) + src_offset,
                PQ * elem_size
            );
        }
    }
}

ZenuStatus ZenuComputeConvCpuImpl::backward_kernel(
    const void* input,
    const void* grad_output,
    void*       grad_kernel,
    void*       workspace
) const {
    if (!input || !grad_output || !grad_kernel || !workspace) {
        return ZenuStatus::InvalidArgument;
    }

    size_t kernel_size = 1;
    for (auto &kdim : kernel) {
        kernel_size *= kdim;
    }
    size_t dtype_size;
    DEFINE_DATA_SIZE(type, dtype_size);
    memset(grad_kernel, 0, kernel_size * dtype_size);

    memset(workspace, 0, get_backward_kernel_bytes());
    im2col(input, workspace);

    const size_t im2col_bytes = get_im2col_bytes();
    char* dY_trans = static_cast<char*>(workspace) + im2col_bytes + 1024;

    std::uintptr_t addr    = reinterpret_cast<std::uintptr_t>(dY_trans);
    std::uintptr_t aligned = (addr + 63) & ~static_cast<std::uintptr_t>(63);
    dY_trans = reinterpret_cast<char*>(aligned);

    transpose_gemm2d_bkwd_kernel(grad_output, dY_trans);

    auto [M, Kdim, N] = get_gemm_param_bkwd_kernel();

    zenu_compute_gemm_cpu(
        NoTranspose, 
        NoTranspose,
        (int)M,
        (int)N,
        (int)Kdim,
        1.0,
        dY_trans, 
        (int)Kdim,
        workspace, 
        (int)N,
        0.0,
        grad_kernel, 
        (int)N,
        type
    );

    return ZenuStatus::Success;
}

size_t ZenuComputeConvCpuImpl::get_backward_kernel_bytes() const {
    const size_t im2col_sz = get_im2col_bytes(); 

    auto [M, Kdim, N] = get_gemm_param_bkwd_kernel();
    size_t dY_trans_elm = M * Kdim;

    size_t dtype_size;
    DEFINE_DATA_SIZE(type, dtype_size);

    return im2col_sz + dY_trans_elm * dtype_size + 2048; 
}


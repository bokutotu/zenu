#include "conv.h"
#include "zenu_compute_blas.h"
#include <cstring>
#include <iostream>

std::array<size_t, 3> ZenuComputeConvCpuImpl::get_gemm_param() const {
    const size_t dim = get_dim();
    if (dim == 2) {
        return get_gemm_param2d();
    }
    // TODO: Support other dimensions
    return {0, 0, 0};
}

std::array<size_t, 3> ZenuComputeConvCpuImpl::get_gemm_param2d() const {
    // M: 出力チャネル数 (K)
    // K: 入力チャネル数×カーネル高さ×カーネル幅 (C*R*S)
    // N: バッチサイズ×出力高さ×出力幅 (N*P*Q)
    const size_t M = output[1];
    const size_t K = input[1] * kernel[2] * kernel[3];
    const size_t N = input[0] * output[2] * output[3];  // Include batch size
    return {M, K, N};
}

ZenuStatus ZenuComputeConvCpuImpl::forward(
    const void* input, 
    const void* kernel, 
    void* output,
    void* workspace
) const {
    if (type != ZenuDataType::f32 && type != ZenuDataType::f64) {
        std::cout << "Unsupported data type in forward" << std::endl;
        return ZenuStatus::InvalidArgument;
    }
    memset(workspace, 0, get_im2col_bytes());
    im2col(input, workspace);

    const size_t im2col_bytes = get_im2col_bytes();
    char* gemm_output = static_cast<char*>(workspace) + im2col_bytes + 1024;
    gemm_output -= reinterpret_cast<size_t>(gemm_output) % 64;

    const auto [M, K, N] = get_gemm_param();
    ZenuStatus status;
    // zenu_compute_gemm_cpu(
    //     NoTranspose,
    //     NoTranspose,
    //     M, N, K,
    //     1.0,
    //     kernel, K,
    //     workspace, N,
    //     0.0,
    //     gemm_output, N,
    //     type
    // );
// いま M = out_ch,  K = in_ch*R*S,  N = batch*P*Q
    zenu_compute_gemm_cpu(
        NoTranspose,
        Transpose,   // ★こちらを Transpose に
        M, N, K,                    // => M x K * K x N = M x N
        1.0,
        kernel,  // A
        K,       // lda = K  (Aは(M,K)だから行幅はK)
        workspace, 
        K,       // ★ldb = K (Bは実メモリで(N,K)だから行幅はK)
        0.0,
        gemm_output, 
        N, 
        type
    );


    transpose_gemm(gemm_output, output);

    return ZenuStatus::Success;
}


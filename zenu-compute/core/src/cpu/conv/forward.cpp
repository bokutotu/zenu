#include "conv.h"
#include "zenu_compute_blas.h"
#include <cstring>
#include <iostream>

std::array<size_t, 3> ZenuComputeConvCpuImpl::get_gemm_param_fwd() const {
    const size_t dim = get_dim();
    if (dim == 2) {
        return get_gemm_param2d_fwd();
    }
    // TODO: Support other dimensions
    return {0, 0, 0};
}

std::array<size_t, 3> ZenuComputeConvCpuImpl::get_gemm_param2d_fwd() const {
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

    const auto [M, K, N] = get_gemm_param_fwd();
    ZenuStatus status;

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


    transpose_gemm_fwd(gemm_output, output);

    return ZenuStatus::Success;
}

void ZenuComputeConvCpuImpl::transpose_gemm_fwd(const void* gemm_out, void* output) const {
    if (get_dim() == 2) {
        transpose_gemm2d_fwd(gemm_out, output);
    } else {
        std::cout << "Unsupported dimension in transpose_gemm" << std::endl;
    }
}

void ZenuComputeConvCpuImpl::transpose_gemm2d_fwd(const void* gemm_out, void* output) const {
    const size_t N = this->output[0];
    const size_t K = this->output[1];
    const size_t P = this->output[2];
    const size_t Q = this->output[3];
    const size_t M = K;
    const size_t spatial_size = P * Q;

#define TRANSPOSE_GEMM_2D_LOOP(TYPE, in_ptr, out_ptr)                    \
    _Pragma("omp parallel for collapse(3)")                             \
    for(size_t n = 0; n < N; ++n) {                                     \
        for(size_t k = 0; k < K; ++k) {                                 \
            for(size_t pq = 0; pq < spatial_size; ++pq) {               \
                ((TYPE*)out_ptr)[n*K*spatial_size + k*spatial_size + pq] = \
                    ((const TYPE*)in_ptr)[k*(N*spatial_size) + n*spatial_size + pq]; \
            }                                                           \
        }                                                               \
    }

    switch (type) {
    case ZenuDataType::f32:
        TRANSPOSE_GEMM_2D_LOOP(float, gemm_out, output);
        break;
    case ZenuDataType::f64:
        TRANSPOSE_GEMM_2D_LOOP(double, gemm_out, output);
        break;
    default:
        std::cout << "Unsupported data type in transpose_gemm2d" << std::endl;
        break;
    }

#undef TRANSPOSE_GEMM_2D_LOOP
}

size_t ZenuComputeConvCpuImpl::get_im2col2d_bytes() const {
    const size_t N           = this->input[0];
    const size_t C           = this->input[1];
    const size_t H           = this->input[2];
    const size_t W           = this->input[3];

    const size_t kernel_h    = this->kernel[2];
    const size_t kernel_w    = this->kernel[3];

    const size_t stride_h    = this->stride[0];
    const size_t stride_w    = this->stride[1];
    const size_t pad_h       = this->padding[0];
    const size_t pad_w       = this->padding[1];
    const size_t dilation_h  = this->dilation[0];
    const size_t dilation_w  = this->dilation[1];

    const size_t out_h       = this->output[2];
    const size_t out_w       = this->output[3];

    size_t data_size;
    switch (type) {
    case ZenuDataType::f32:
        data_size = sizeof(float);
        break;
    case ZenuDataType::f64:
        data_size = sizeof(double);
        break;
    default:
        std::cout << "Unsupported data type" << std::endl;
        exit(1);
    }

    return N * C * kernel_h * kernel_w * out_h * out_w * data_size;
}

size_t ZenuComputeConvCpuImpl::get_im2col_bytes() const {
    if (get_dim() == 2) {
        return get_im2col2d_bytes();
    } else {
        return 0;
    }
}

size_t ZenuComputeConvCpuImpl::get_gemm_bytes_fwd() const {
    size_t num_elm = 1;
    for (int i = 0; i < output.size(); i++) {
        num_elm *= output[i];
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
    return num_elm * data_bytes;
}

size_t ZenuComputeConvCpuImpl::get_forward_bytes() const {
    return get_im2col_bytes() + get_gemm_bytes_fwd() + 1024;
}

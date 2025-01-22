#include "conv.h"
#include "zenu_compute_type.h"

#include <omp.h>
#include <iostream>

void ZenuComputeConvCpuImpl::transpose_gemm(const void* gemm_out, void* output) const {
    if (get_dim() == 2) {
        transpose_gemm2d(gemm_out, output);
    } else {
        std::cout << "Unsupported dimension in transpose_gemm" << std::endl;
    }
}

void ZenuComputeConvCpuImpl::transpose_gemm2d(const void* gemm_out, void* output) const {
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


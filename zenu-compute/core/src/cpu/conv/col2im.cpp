#include "conv.h"
#include <omp.h>
#include <iostream>

/**
 * @brief 2D Conv向け col2im 実装例
 * 
 * im2col2d で生成した col バッファ(形状: [N*P*Q, C*R*S]) を
 * 入力勾配 (N,C,H,W) 形式に加算します (+=)。
 */
void ZenuComputeConvCpuImpl::col2im2d(const void* col, void* input) const
{
    // 入力形状
    const size_t N = this->input[0];
    const size_t C = this->input[1];
    const size_t H = this->input[2];
    const size_t W = this->input[3];

    // カーネル形状
    const size_t kernel_h = this->kernel[2];
    const size_t kernel_w = this->kernel[3];

    // ストライド・パディングなど
    const size_t stride_h   = this->stride[0];
    const size_t stride_w   = this->stride[1];
    const size_t pad_h      = this->padding[0];
    const size_t pad_w      = this->padding[1];
    const size_t dilation_h = this->dilation[0];
    const size_t dilation_w = this->dilation[1];

    // 出力空間サイズ (forwardのConv2Dでいう P,Q)
    const size_t out_h = this->output[2];
    const size_t out_w = this->output[3];

#define COL2IM_2D_LOOP(TYPE)                                           \
    do {                                                               \
        TYPE* in_ptr = static_cast<TYPE*>(input);                      \
        const TYPE* col_ptr = static_cast<const TYPE*>(col);           \
        /* OpenMP並列化(必要に応じて粒度を調整) */                      \
        _Pragma("omp parallel for collapse(2)")                        \
        for (size_t n = 0; n < N; ++n) {                               \
            for (size_t c = 0; c < C; ++c) {                           \
                for (size_t oh = 0; oh < out_h; ++oh) {                \
                    for (size_t ow = 0; ow < out_w; ++ow) {            \
                        for (size_t kh = 0; kh < kernel_h; ++kh) {     \
                            for (size_t kw = 0; kw < kernel_w; ++kw) { \
                                const int h_in =                       \
                                    static_cast<int>(oh * stride_h + kh * dilation_h - pad_h); \
                                const int w_in =                       \
                                    static_cast<int>(ow * stride_w + kw * dilation_w - pad_w); \
                                if (h_in >= 0 && h_in < (int)H &&       \
                                    w_in >= 0 && w_in < (int)W) {       \
                                    /* col上のインデックス */          \
                                    const size_t col_index =           \
                                        ((n * out_h * out_w) + (oh * out_w) + ow) * (C * kernel_h * kernel_w) \
                                        + (c * kernel_h * kernel_w)     \
                                        + (kh * kernel_w)               \
                                        + kw;                           \
                                    /* input勾配のインデックス */       \
                                    const size_t in_index =            \
                                        ((n * C + c) * H + (size_t)h_in) * W + (size_t)w_in; \
                                    in_ptr[in_index] += col_ptr[col_index]; \
                                } \
                            } \
                        } \
                    } \
                } \
            } \
        }                                                               \
    } while (0)

    switch (this->type) {
    case ZenuDataType::f32:
        COL2IM_2D_LOOP(float);
        break;
    case ZenuDataType::f64:
        COL2IM_2D_LOOP(double);
        break;
    default:
        std::cout << "Unsupported data type in col2im2d" << std::endl;
        break;
    }

#undef COL2IM_2D_LOOP
}

/**
 * @brief 次元に応じて col2im2d を呼び分け
 */
void ZenuComputeConvCpuImpl::col2im(const void* col, void* input) const
{
    if (get_dim() == 2) {
        col2im2d(col, input);
    } else {
        std::cout << "col2im: Unsupported dimension" << std::endl;
    }
}


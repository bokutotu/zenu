#include "conv.h"

#include <omp.h>
#include <iostream>
#include <cstddef>

void ZenuComputeConvCpuImpl::im2col2d(const void* input, void* col) const {
    // 入力形状
    const size_t N = this->input[0];
    const size_t C = this->input[1];
    const size_t H = this->input[2];
    const size_t W = this->input[3];

    // カーネル形状
    const size_t kernel_h = this->kernel[2];
    const size_t kernel_w = this->kernel[3];

    // ストライドやパディングなど
    const size_t stride_h   = this->stride[0];
    const size_t stride_w   = this->stride[1];
    const size_t pad_h      = this->padding[0];
    const size_t pad_w      = this->padding[1];
    const size_t dilation_h = this->dilation[0];
    const size_t dilation_w = this->dilation[1];

    // 出力形状（Conv2D後）
    const size_t out_h = this->output[2];
    const size_t out_w = this->output[3];

    #define IM2COL_2D_LOOP(TYPE, ZERO_VAL) \
        TYPE* col_ptr = static_cast<TYPE*>(col); \
        const TYPE* in_ptr = static_cast<const TYPE*>(input); \
        _Pragma("omp parallel for collapse(3)") \
        for (size_t n = 0; n < N; ++n) { \
            for (size_t oh = 0; oh < out_h; ++oh) { \
                for (size_t ow = 0; ow < out_w; ++ow) { \
                    for (size_t c = 0; c < C; ++c) { \
                        for (size_t kh = 0; kh < kernel_h; ++kh) { \
                            for (size_t kw = 0; kw < kernel_w; ++kw) { \
                                const int h_in = static_cast<int>(oh * stride_h + kh * dilation_h - pad_h); \
                                const int w_in = static_cast<int>(ow * stride_w + kw * dilation_w - pad_w); \
                                const size_t col_index = \
                                    ((n * out_h * out_w) + (oh * out_w) + ow) * (C * kernel_h * kernel_w) \
                                    + (c * kernel_h * kernel_w) \
                                    + (kh * kernel_w) \
                                    + kw; \
                                if (h_in >= 0 && h_in < static_cast<int>(H) && w_in >= 0 && w_in < static_cast<int>(W)) { \
                                    const size_t in_index = \
                                        (n * C + c) * (H * W) \
                                        + (static_cast<size_t>(h_in) * W) \
                                        + static_cast<size_t>(w_in); \
                                    col_ptr[col_index] = in_ptr[in_index]; \
                                } else { \
                                    col_ptr[col_index] = ZERO_VAL; \
                                } \
                            } \
                        } \
                    } \
                } \
            } \
        }

    switch (this->type) {
        case ZenuDataType::f32: {
            IM2COL_2D_LOOP(float, 0.f);
            break;
        }
        case ZenuDataType::f64: {
            IM2COL_2D_LOOP(double, 0.0);
            break;
        }
        default: {
            std::cout << "Unsupported data type in im2col2d" << std::endl;
            break;
        }
    }

    #undef IM2COL_2D_LOOP
}

void ZenuComputeConvCpuImpl::im2col(const void* input, void* col) const {
    // 2次元Convのみ対応
    if (get_dim() == 2) {
        im2col2d(input, col);
    } else {
        std::cout << "Unsupported dimension" << std::endl;
    }
}


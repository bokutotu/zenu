#include "conv.h"

#include <omp.h>
#include <iostream>

void ZenuComputeConvCpu::im2col2d(const void* input, void* col) const {
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

#define IM2COL_2D_LOOP(TYPE, ZERO_VAL, in_ptr, col_ptr,       \
                       N, C, H, W,                            \
                       KERNEL_H, KERNEL_W,                    \
                       STRIDE_H, STRIDE_W,                    \
                       PAD_H, PAD_W,                          \
                       DIL_H, DIL_W,                          \
                       OUT_H, OUT_W)                          \
    do {                                                      \
        _Pragma("omp parallel for collapse(2)")               \
        for (size_t n = 0; n < (N); n++) {                     \
            for (size_t c = 0; c < (C); c++) {                 \
                for (size_t kh = 0; kh < (KERNEL_H); kh++) {   \
                    for (size_t kw = 0; kw < (KERNEL_W); kw++) {\
                        for (size_t oh = 0; oh < (OUT_H); oh++) {\
                            for (size_t ow = 0; ow < (OUT_W); ow++) {\
                                int h_in = static_cast<int>(oh * (STRIDE_H) + kh * (DIL_H) - (PAD_H)); \
                                int w_in = static_cast<int>(ow * (STRIDE_W) + kw * (DIL_W) - (PAD_W)); \
                                size_t col_index =                                                 \
                                    n * ((C) * (KERNEL_H) * (KERNEL_W) * (OUT_H) * (OUT_W))          \
                                    + c * ((KERNEL_H) * (KERNEL_W) * (OUT_H) * (OUT_W))              \
                                    + (kh * (KERNEL_W) + kw) * ((OUT_H) * (OUT_W))                   \
                                    + oh * (OUT_W) + ow;                                             \
                                if (h_in >= 0 && h_in < static_cast<int>(H) &&                        \
                                    w_in >= 0 && w_in < static_cast<int>(W)) {                        \
                                    size_t in_index =                                                \
                                        n * ((C) * (H) * (W))                                         \
                                        + c * ((H) * (W))                                             \
                                        + (static_cast<size_t>(h_in) * (W))                           \
                                        + static_cast<size_t>(w_in);                                  \
                                    ((TYPE*)(col_ptr))[col_index] =                                  \
                                        ((const TYPE*)(in_ptr))[in_index];                            \
                                } else {                                                              \
                                    ((TYPE*)(col_ptr))[col_index] = (ZERO_VAL);                      \
                                }                                                                     \
                            }                                                                         \
                        }                                                                             \
                    }                                                                                 \
                }                                                                                     \
            }                                                                                         \
        }                                                                                             \
    } while (0)


    switch (this->type)
    {
    case f32:
        IM2COL_2D_LOOP(float, 0.f, input, col,
                       N, C, H, W,
                       kernel_h, kernel_w,
                       stride_h, stride_w,
                       pad_h, pad_w,
                       dilation_h, dilation_w,
                       out_h, out_w);
        break;

    case f64:
        IM2COL_2D_LOOP(double, 0.0, input, col,
                       N, C, H, W,
                       kernel_h, kernel_w,
                       stride_h, stride_w,
                       pad_h, pad_w,
                       dilation_h, dilation_w,
                       out_h, out_w);
        break;

    default:
        std::cout << "Unsupported data type" << std::endl;
        break;
    }

#undef IM2COL_2D_LOOP
}

void ZenuComputeConvCpu::im2col(const void* input, void* col) const {
    if (get_dim() == 2) {
        im2col2d(input, col);
    } else {
        std::cout << "Unsupported dimension" << std::endl;
    }
}

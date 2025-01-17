#include <cstring>
#include <omp.h>
#include <cstddef>
#include "zenu_compute_arithmetic.h"

/**
 * @brief バッチ次元も含めて、入力 (NCHW) を巨大な2次元行列 (im2col) に変換する。
 * 
 * 出力形状:  
 *   [ out_rows, out_cols ] (row-major)
 *   where
 *     out_rows = channels * kernel_h * kernel_w
 *     out_cols = batch_size * height_out * width_out
 * 
 * @tparam T (float や double など)
 * @param[in] input    入力配列 (形状: [N, C, H_in, W_in])
 * @param[in] batch_size
 * @param[in] channels
 * @param[in] height_in
 * @param[in] width_in
 * @param[in] height_out
 * @param[in] width_out
 * @param[in] kernel_h
 * @param[in] kernel_w
 * @param[in] pad_h
 * @param[in] pad_w
 * @param[in] stride_h
 * @param[in] stride_w
 * @param[in] dilation_h
 * @param[in] dilation_w
 * @param[out] output  出力先 (2次元行列) 
 */
template <typename T>
void im2col2d(
    const T*    input,
    int         batch_size,
    int         channels,
    int         height_in,
    int         width_in,
    int         height_out,
    int         width_out,
    int         kernel_h,
    int         kernel_w,
    int         pad_h,
    int         pad_w,
    int         stride_h,
    int         stride_w,
    int         dilation_h,
    int         dilation_w,
    T*          output
)
{
    const int out_rows = channels * kernel_h * kernel_w;
    const int out_cols = batch_size * height_out * width_out;

    float zero = 0.;
    zenu_compute_mul_mat_scalar_ptr_assign_cpu(
        output,
        1,
        &zero,
        out_rows * out_cols,
        ZenuDataType::f32
    );

#pragma omp parallel for collapse(3)
    for(int c = 0; c < channels; ++c){
        for(int kh = 0; kh < kernel_h; ++kh){
            for(int kw = 0; kw < kernel_w; ++kw){
                const int row = (c * kernel_h + kh) * kernel_w + kw;

                for(int n = 0; n < batch_size; ++n){
                    for(int oh = 0; oh < height_out; ++oh){
                        const int ih = oh * stride_h - pad_h + kh * dilation_h;
                        for(int ow = 0; ow < width_out; ++ow){
                            const int iw = ow * stride_w - pad_w + kw * dilation_w;

                            const int col = n*(height_out*width_out) + oh*width_out + ow;

                            if(ih < 0 || ih >= height_in || iw < 0 || iw >= width_in){
                                output[row * out_cols + col] = static_cast<T>(0);
                            } else {
                                const size_t in_offset = 
                                    static_cast<size_t>(n)*channels*height_in*width_in
                                    + static_cast<size_t>(c)*height_in*width_in
                                    + static_cast<size_t>(ih)*width_in
                                    + iw;

                                output[row * out_cols + col] = input[in_offset];
                            }
                        }
                    }
                }
            }
        }
    }
}


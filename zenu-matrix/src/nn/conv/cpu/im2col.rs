use crate::num::Num;

use super::super::utils::conv_dim_out_size;

/// im2col_all:
/// 入力: NCHW
/// 出力: col (C_in*KH*KW, N*out_h*out_w)
/// バッチ方向Nもまとめて1つの行列に展開する。
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn im2col<T: Num>(
    input: &[T],
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    col: &mut [T],
) {
    let out_h = conv_dim_out_size(h_in, kh, pad_h, stride_h, dilation_h);
    let out_w = conv_dim_out_size(w_in, kw, pad_w, stride_w, dilation_w);

    let rows = c_in * kh * kw;
    let cols = n * out_h * out_w;
    assert_eq!(col.len(), rows * cols);

    for c in 0..c_in {
        for k_h in 0..kh {
            let ih_base = k_h * dilation_h;
            for k_w in 0..kw {
                let iw_base = k_w * dilation_w;
                let row = c * kh * kw + k_h * kw + k_w;
                for ni in 0..n {
                    for oh in 0..out_h {
                        let ih = oh * stride_h + ih_base as usize - pad_h;
                        for ow in 0..out_w {
                            let iw = ow * stride_w + iw_base as usize - pad_w;
                            let col_idx = ni * (out_h * out_w) + oh * out_w + ow;
                            col[row * cols + col_idx] = if ih < h_in && iw < w_in {
                                input
                                    [ni * (c_in * h_in * w_in) + c * (h_in * w_in) + ih * w_in + iw]
                            } else {
                                T::zero()
                            };
                        }
                    }
                }
            }
        }
    }
}

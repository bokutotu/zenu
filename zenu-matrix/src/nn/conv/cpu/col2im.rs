use crate::num::Num;

use super::super::utils::conv_dim_out_size;

#[expect(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
pub fn col2im<T: Num>(
    col: &[T],
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
    dx: &mut [T],
) {
    let out_h = conv_dim_out_size(h_in, kh, pad_h, stride_h, dilation_h);
    let out_w = conv_dim_out_size(w_in, kw, pad_w, stride_w, dilation_w);

    let rows = c_in * kh * kw;
    let cols = n * out_h * out_w;
    assert_eq!(col.len(), rows * cols);

    for v in dx.iter_mut() {
        *v = T::zero();
    }

    for c in 0..c_in {
        for k_h in 0..kh {
            let ih_base = k_h * dilation_h;
            for k_w in 0..kw {
                let iw_base = k_w * dilation_w;
                let row = c * kh * kw + k_h * kw + k_w;
                for ni in 0..n {
                    for oh in 0..out_h {
                        let ih_ = (oh * stride_h) as isize + ih_base as isize - pad_h as isize;
                        if ih_ < 0 || ih_ >= h_in as isize {
                            continue;
                        }
                        let ih = ih_ as usize;

                        for ow in 0..out_w {
                            let iw_ = (ow * stride_w) as isize + iw_base as isize - pad_w as isize;
                            if iw_ < 0 || iw_ >= w_in as isize {
                                continue;
                            }
                            let iw = iw_ as usize;

                            let col_idx = ni * (out_h * out_w) + oh * out_w + ow;
                            dx[ni * (c_in * h_in * w_in) + c * (h_in * w_in) + ih * w_in + iw] +=
                                col[row * cols + col_idx];
                        }
                    }
                }
            }
        }
    }
}

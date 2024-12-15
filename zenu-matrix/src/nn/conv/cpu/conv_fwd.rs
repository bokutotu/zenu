use crate::{
    device::cpu::Cpu, matrix_blas::BlasTrans, nn::conv::utils::conv_dim_out_size, num::Num,
    operation::mul::Gemm,
};

use super::im2col::im2col;

#[expect(clippy::too_many_arguments)]
pub fn conv_fwd<T: Num>(
    input: &[T],
    filter: &[T],
    n: usize,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    output: &mut [T],
) {
    let out_h = conv_dim_out_size(h_in, kh, pad_h, stride_h, dilation_h);
    let out_w = conv_dim_out_size(w_in, kw, pad_w, stride_w, dilation_w);

    let col_size = c_in * kh * kw * n * out_h * out_w;
    let mut col = vec![T::zero(); col_size];
    im2col(
        input, n, c_in, h_in, w_in, kh, kw, pad_h, pad_w, stride_h, stride_w, dilation_h,
        dilation_w, &mut col,
    );

    // out_mat = filter * col
    let mut out_mat = vec![T::zero(); c_out * n * out_h * out_w];

    Cpu::gemm_unchecked(
        BlasTrans::None,
        BlasTrans::None,
        c_out,
        n * out_h * out_w,
        c_in * kh * kw,
        T::one(),
        filter.as_ptr(),
        c_in * kh * kw,
        col.as_ptr(),
        n * out_h * out_w,
        T::zero(),
        out_mat.as_mut_ptr(),
        n * out_h * out_w,
    );

    // reshape out_mat to NCHW
    for ni in 0..n {
        for co in 0..c_out {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_idx = ni * c_out * out_h * out_w + co * out_h * out_w + oh * out_w + ow;
                    let mat_idx = co * (n * out_h * out_w) + ni * (out_h * out_w) + oh * out_w + ow;
                    output[out_idx] = out_mat[mat_idx];
                }
            }
        }
    }
}

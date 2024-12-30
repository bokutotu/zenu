use crate::{
    device::cpu::Cpu, matrix_blas::BlasTrans, nn::conv::utils::conv_dim_out_size, num::Num,
    operation::mul::Gemm,
};

use super::col2im::col2im;

#[expect(clippy::too_many_arguments)]
pub fn conv_bkwd_data<T: Num>(
    dy: &[T],
    filter: &[T],
    n: usize,
    c_in: usize,
    c_out: usize,
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

    let nn = n * out_h * out_w;
    let mut dy_mat = vec![T::zero(); c_out * nn];
    for ni in 0..n {
        for co in 0..c_out {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let dy_idx = ni * c_out * out_h * out_w + co * out_h * out_w + oh * out_w + ow;
                    let mat_idx = co * nn + ni * (out_h * out_w) + oh * out_w + ow;
                    dy_mat[mat_idx] = dy[dy_idx];
                }
            }
        }
    }

    let m = c_in * kh * kw;
    let k = c_out;
    let ncol = n * out_h * out_w;
    let mut col = vec![T::zero(); m * ncol];

    Cpu::gemm_unchecked(
        BlasTrans::Ordinary,
        BlasTrans::None,
        m,
        ncol,
        k,
        T::one(),
        filter.as_ptr(),
        c_in * kh * kw,
        dy_mat.as_ptr(),
        ncol,
        T::zero(),
        col.as_mut_ptr(),
        ncol,
    );

    col2im(
        &col, n, c_in, h_in, w_in, kh, kw, pad_h, pad_w, stride_h, stride_w, dilation_h,
        dilation_w, dx,
    );
}

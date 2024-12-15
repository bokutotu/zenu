use crate::{
    device::cpu::Cpu, matrix_blas::BlasTrans, nn::conv::utils::conv_dim_out_size, num::Num,
    operation::mul::Gemm,
};

use super::im2col::im2col;

#[expect(clippy::many_single_char_names, clippy::too_many_arguments)]
pub fn conv_bkwd_filter<T: Num>(
    dy: &[T],
    x: &[T],
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
    dw: &mut [T],
) {
    let out_h = conv_dim_out_size(h_in, kh, pad_h, stride_h, dilation_h);
    let out_w = conv_dim_out_size(w_in, kw, pad_w, stride_w, dilation_w);

    let col_size = c_in * kh * kw * n * out_h * out_w;
    let mut col = vec![T::zero(); col_size];
    im2col(
        x, n, c_in, h_in, w_in, kh, kw, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        &mut col,
    );

    let mut dy_mat = vec![T::zero(); c_out * n * out_h * out_w];
    for ni in 0..n {
        for co in 0..c_out {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let dy_idx = ni * c_out * out_h * out_w + co * out_h * out_w + oh * out_w + ow;
                    let mat_idx = co * (n * out_h * out_w) + ni * (out_h * out_w) + oh * out_w + ow;
                    dy_mat[mat_idx] = dy[dy_idx];
                }
            }
        }
    }

    // dw = dy_mat * col^T
    // ここでtransb = BlasTrans::Ordinaryを使用して転置を内部的に指示し、forループでの転置を省略
    let m = c_out;
    let k = n * out_h * out_w;
    let p = c_in * kh * kw;

    let mut dw_temp = vec![T::zero(); c_out * c_in * kh * kw];

    Cpu::gemm_unchecked(
        BlasTrans::None,     // dy_matはそのまま
        BlasTrans::Ordinary, // colを転置扱い
        m,
        p,
        k,
        T::one(),
        dy_mat.as_ptr(),
        k,
        col.as_ptr(),
        k, // colは( C_in*KH*KW, N*out_h*out_w ) => 転置すれば(N*out_h*out_w, C_in*KH*KW)
        T::zero(),
        dw_temp.as_mut_ptr(),
        p,
    );

    dw.copy_from_slice(&dw_temp);
}

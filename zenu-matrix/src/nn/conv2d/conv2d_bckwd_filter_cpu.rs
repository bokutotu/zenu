use crate::{
    device::Device,
    dim::DimDyn,
    matrix::{Matrix, Owned, Ref},
    num::Num,
    operation::mul::matmul,
};

use super::im2col::{im2col, Im2ColRes};

pub(super) fn conv2d_bckwd_fileter<T: Num, D: Device>(
    input: Matrix<Ref<&T>, DimDyn, D>,
    kernel: Matrix<Ref<&T>, DimDyn, D>,
    gradient_output: Matrix<Ref<&T>, DimDyn, D>,
    padding: (usize, usize),
    stride: (usize, usize),
) -> Matrix<Owned<T>, DimDyn, D> {
    let kernel_shape = kernel.shape();
    let kernel_h_w = (kernel_shape[2], kernel_shape[3]);
    let Im2ColRes { mut col, .. } = im2col(input, kernel_h_w, stride, padding);
    col.transpose();
    let gradient_output = gradient_output.transpose_swap_index_new_matrix(0, 1);
    let gradient_output_shape = gradient_output.shape();
    let gradient_output = gradient_output.reshape([
        gradient_output_shape[0],
        gradient_output_shape[1] * gradient_output_shape[2] * gradient_output_shape[3],
    ]);

    matmul(&gradient_output, &col)
}

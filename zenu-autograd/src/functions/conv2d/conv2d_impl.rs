use zenu_matrix::{
    constructor::zeros::Zeros,
    dim::DimDyn,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::{Matrix, OwnedMatrixDyn},
    memory_impl::{OwnedMem, ViewMem},
    num::Num,
    operation::{mul::Gemm, reshape::Reshape, transpose::TransposeInplace},
};

use super::im2col::{im2col, Im2ColRes};

pub(crate) fn conv2d_inner<T: Num>(
    img: Matrix<ViewMem<T>, DimDyn>,
    kernel: Matrix<ViewMem<T>, DimDyn>,
    padding: (usize, usize),
    stride: (usize, usize),
) -> Matrix<OwnedMem<T>, DimDyn> {
    let batch_size = img.shape()[0];
    let kernel_shape = kernel.shape();
    let kernel_h_w = (kernel_shape[1], kernel_shape[2]);
    let Im2ColRes { col, out_size } = im2col(img, kernel_h_w, stride, padding);
    // col shape -> [n * oh * ow, c * kh * kw]
    // kernel shape -> [c, kh, kw, out_channelsk]
    let kernel = kernel.reshape([col.shape()[1], kernel_shape[3]]);
    let mut out = OwnedMatrixDyn::zeros([col.shape()[0], kernel_shape[3]]);
    out.to_view_mut().gemm(col.to_view(), kernel.to_view());
    out.reshape([batch_size, out_size.0, out_size.1, kernel_shape[3]])
        .transpose_by_index_inplace(&[3, 0, 1, 2])
}

#[cfg(test)]
mod conv2d {
    use zenu_matrix::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::asum::Asum,
    };

    use super::conv2d_inner;

    #[test]
    fn conv2d_5x5im_3x3_kernel_0x0_pad_1x1_stride() {
        let kernel =
            OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], [1, 3, 3, 1]);
        let img = (1..26).map(|x| x as f32).collect::<Vec<f32>>();
        let img = OwnedMatrixDyn::from_vec(img, [1, 1, 5, 5]);
        let out = conv2d_inner(img.to_view(), kernel.to_view(), (0, 0), (1, 1));
        let ans = OwnedMatrixDyn::from_vec(
            vec![411., 456., 501., 636., 681., 726., 861., 906., 951.],
            [1, 1, 3, 3],
        );
        assert!((out - ans).asum() < 1e-6);
    }
}

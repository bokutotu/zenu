use zenu_matrix::{
    constructor::zeros::Zeros,
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, MatrixSliceDyn, MatrixSliceMutDyn, ToViewMutMatrix},
    matrix_impl::{Matrix, OwnedMatrixDyn},
    memory_impl::{OwnedMem, ViewMem},
    num::Num,
    operation::{copy_from::CopyFrom, reshape::Reshape, transpose::TransposeInplace},
    slice_dynamic,
};

pub fn padding<T: Num>(
    input: Matrix<ViewMem<T>, DimDyn>,
    padding: (usize, usize),
) -> Matrix<OwnedMem<T>, DimDyn> {
    let (padding_height, padding_width) = padding;
    let (batch_size, in_channels, in_height, in_width) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let out_height = in_height + 2 * padding_height;
    let out_width = in_width + 2 * padding_width;

    let mut output = OwnedMatrixDyn::zeros([batch_size, in_channels, out_height, out_width]);
    let mut output_view_mut = output.to_view_mut();

    let mut output_view_mut = output_view_mut.slice_mut_dyn(slice_dynamic!(
        ..,
        ..,
        padding_height..padding_height + in_height,
        padding_width..padding_width + in_width
    ));
    output_view_mut.copy_from(&input);

    output
}

pub(crate) struct Im2ColRes<T: Num> {
    pub(crate) col: Matrix<OwnedMem<T>, DimDyn>,
    pub(crate) out_size: (usize, usize),
}

pub(crate) fn im2col<T: Num>(
    img: Matrix<ViewMem<T>, DimDyn>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
) -> Im2ColRes<T> {
    let n = img.shape()[0];
    let c = img.shape()[1];
    let h = img.shape()[2];
    let w = img.shape()[3];
    let kh = kernel_size.0;
    let kw = kernel_size.1;
    let sh = stride.0;
    let sw = stride.1;
    let ph = pad.0;
    let pw = pad.1;
    let oh = (h + 2 * ph - kh) / stride.0 + 1;
    let ow = (w + 2 * pw - kw) / stride.1 + 1;

    let img = padding(img, pad);
    let mut col = OwnedMatrixDyn::zeros([n, c, kh, kw, oh, ow]);

    for j in 0..kh {
        let j_lim = j + sh * oh;
        for i in 0..kw {
            let i_lim = i + sw * ow;
            let mut col = col.slice_mut_dyn(slice_dynamic!(.., .., j, i, .., ..));
            let img = img.slice_dyn(slice_dynamic!(.., .., j..j_lim;sh, i..i_lim;sw));
            col.copy_from(&img);
        }
    }

    let num_elm = col.shape().num_elm();
    let col = col.transepose_by_index(&[0, 4, 5, 1, 2, 3]);
    let col = col.reshape_new_matrix(&[n * oh * ow, num_elm / (n * oh * ow)]);
    Im2ColRes {
        col,
        out_size: (oh, ow),
    }
}

#[cfg(test)]
mod im2col {
    use zenu_matrix::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::asum::Asum,
    };

    use super::im2col;

    #[test]
    fn im2col_small() {
        let input = OwnedMatrixDyn::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [1, 1, 4, 4],
        );
        let result = im2col(input.to_view(), (2, 2), (1, 1), (0, 0));
        let ans = OwnedMatrixDyn::from_vec(
            vec![
                1., 2., 5., 6., 2., 3., 6., 7., 3., 4., 7., 8., 5., 6., 9., 10., 6., 7., 10., 11.,
                7., 8., 11., 12., 9., 10., 13., 14., 10., 11., 14., 15., 11., 12., 15., 16.,
            ],
            [9, 4],
        );
        assert!((ans - result.col).to_view().asum() < 1e-6);
    }
}

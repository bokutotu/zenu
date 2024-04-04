use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zenu_matrix::{
    constructor::zeros::Zeros,
    dim::DimDyn,
    matrix::{MatrixBase, MatrixSliceDyn, MatrixSliceMutDyn, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::{Matrix, OwnedMatrixDyn},
    memory_impl::{OwnedMem, ViewMem},
    num::Num,
    operation::{
        copy_from::CopyFrom,
        reshape::{Reshape, ReshapeMut},
        transpose::TransposeInplace,
    },
    slice_dynamic,
};

fn padding<T: Num>(
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

pub struct Im2ColRes<T: Num> {
    pub(crate) col: Matrix<OwnedMem<T>, DimDyn>,
    pub(crate) out_size: (usize, usize),
}

pub fn im2col<T: Num>(
    img: Matrix<ViewMem<T>, DimDyn>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
) -> Im2ColRes<T> {
    let batch_size = img.shape()[0];
    let c = img.shape()[1];
    let h = img.shape()[2];
    let w = img.shape()[3];
    let kh = kernel_size.0;
    let kw = kernel_size.1;
    let sh = stride.0;
    let sw = stride.1;
    let ph = pad.0;
    let pw = pad.1;
    let oh = (h - kh + 2 * ph) / sh + 1;
    let ow = (w - kw + 2 * pw) / sw + 1;

    let img = padding(img, pad);
    let mut col = OwnedMatrixDyn::zeros([batch_size, c, kh, kw, oh, ow]);

    for j in 0..kh {
        let j_lim = j + sh * oh;
        for i in 0..kw {
            let i_lim = i + sw * ow;
            let mut col = col.slice_mut_dyn(slice_dynamic!(.., .., j, i, .., ..));
            let img = img.slice_dyn(slice_dynamic!(.., .., j..j_lim;sh, i..i_lim;sw));
            col.copy_from(&img);
        }
    }

    let col = col.reshape_mut([batch_size, c * kh * kw, oh * ow]);
    let col = col.transepose_by_index(&[1, 0, 2]);
    let col = col.reshape_new_matrix([c * kh * kw, batch_size * oh * ow]);
    Im2ColRes {
        col,
        out_size: (oh, ow),
    }
}

fn im2col_bench(c: &mut Criterion) {
    let input = black_box(OwnedMatrixDyn::<f32>::zeros([32, 16, 128, 128]));
    let kernel_size = (3, 3);
    let stride = (1, 1);
    let pad = (1, 1);

    c.bench_function("im2col_function", |b| {
        b.iter(|| {
            let _ = im2col(input.to_view(), kernel_size, stride, pad);
        })
    });
}

fn padding_bench(c: &mut Criterion) {
    let input = black_box(OwnedMatrixDyn::<f32>::zeros([32, 16, 128, 128]));
    let pad = (1, 1);

    c.bench_function("padding_function", |b| {
        b.iter(|| {
            let _ = padding(input.to_view(), pad);
        })
    });
}

criterion_group!(benches, im2col_bench, padding_bench);
criterion_main!(benches);

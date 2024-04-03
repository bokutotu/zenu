use criterion::{criterion_group, criterion_main, Criterion};
use zenu_matrix::{
    constructor::zeros::Zeros,
    dim::DimDyn,
    matrix::{MatrixSliceDyn, MatrixSliceMutDyn, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::{Matrix, OwnedMatrixDyn},
    memory_impl::{ViewMem, ViewMutMem},
    operation::copy_from::CopyFrom,
    slice_dynamic,
};

fn copy_from_(
    mut a: Matrix<ViewMutMem<f32>, DimDyn>,
    b: Matrix<ViewMem<f32>, DimDyn>,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
) {
    for j in 0..kh {
        let j_lim = j + sh * oh;
        for i in 0..kw {
            let i_lim = i + sw * ow;
            let mut a = a.slice_mut_dyn(slice_dynamic!(.., .., j, i, .., ..));
            let b = b.slice_dyn(slice_dynamic!(.., .., j..j_lim;sh, i..i_lim;sw));
            a.copy_from(&b);
        }
    }
}

fn copy_from_im2col_way(c: &mut Criterion) {
    let b = OwnedMatrixDyn::zeros([32, 16, 128, 128]);
    let mut a = OwnedMatrixDyn::zeros([32, 16, 3, 3, 126, 126]);

    let kh = 3;
    let kw = 3;
    let sh = 1;
    let sw = 1;
    let oh = 126;
    let ow = 126;

    c.bench_function("copy_from_im2col_way", |b_| {
        b_.iter(|| copy_from_(a.to_view_mut(), b.to_view(), kh, kw, oh, ow, sh, sw))
    });
}

criterion_group!(benches, copy_from_im2col_way);
criterion_main!(benches);

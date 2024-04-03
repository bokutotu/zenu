use criterion::{criterion_group, criterion_main, Criterion};
use zenu_matrix::{
    constructor::zeros::Zeros,
    dim::DimDyn,
    matrix::{ToViewMatrix, ToViewMutMatrix},
    matrix_impl::{Matrix, OwnedMatrixDyn},
    memory_impl::{ViewMem, ViewMutMem},
    operation::copy_from::CopyFrom,
};

fn copy_from_all_matrix(mut a: Matrix<ViewMutMem<f32>, DimDyn>, b: Matrix<ViewMem<f32>, DimDyn>) {
    a.copy_from(&b);
}

fn copy_from_all_matrix_(c: &mut Criterion) {
    let b = OwnedMatrixDyn::zeros([32, 16, 128, 128]);
    let mut a = OwnedMatrixDyn::zeros([32, 16, 128, 128]);

    c.bench_function("copy_from_all_matrix", |b_| {
        b_.iter(|| copy_from_all_matrix(a.to_view_mut(), b.to_view()))
    });
}

criterion_group!(benches, copy_from_all_matrix_);
criterion_main!(benches);

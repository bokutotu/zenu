use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zenu_matrix::{
    constructor::zeros::Zeros,
    dim::DimDyn,
    matrix::ToViewMatrix,
    matrix_impl::{Matrix, OwnedMatrixDyn},
    memory_impl::ViewMem,
    operation::{reshape::Reshape, transpose::TransposeInplace},
};

fn transpose_reshape(a: Matrix<ViewMem<f32>, DimDyn>) {
    // let a = a.transepose_by_index(&[1, 2, 3, 0, 4, 5]);
    // let _ = a.reshape_new_matrix([16 * 3 * 3, 32 * 126 * 126]);
    let a = a.reshape([32, 3 * 3 * 16, 126 * 126]);
    let a = a.transpose_by_index_inplace(&[1, 0, 2]);
    let _ = a.reshape([16 * 3 * 3, 32 * 126 * 126]);
}

fn bench(c: &mut Criterion) {
    let a = black_box(OwnedMatrixDyn::zeros([32, 16, 3, 3, 126, 126]));
    // let a = OwnedMatrixDyn::zeros([32, 16 * 3 * 3, 126 * 126]);
    c.bench_function("transpose_reshape_im2col", |b| {
        b.iter(|| transpose_reshape(a.to_view()))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);

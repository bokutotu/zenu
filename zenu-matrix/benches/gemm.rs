use zenu_matrix::{
    dim::DimDyn,
    matrix_impl::{Matrix, OwnedMatrixDyn},
    memory_impl::{ViewMem, ViewMutMem},
    operation::mul::Gemm,
};

fn gemm_func(
    a: Matrix<ViewMem<f32>, DimDyn>,
    b: Matrix<ViewMem<f32>, DimDyn>,
    c: Matrix<ViewMutMem<f32>, DimDyn>,
) {
    c.gemm(a, b);
}

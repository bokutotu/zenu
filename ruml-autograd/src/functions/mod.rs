use ruml_matrix::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::{ViewMemory, ViewMutMemory},
    num::Num,
    operation::{copy_from::CopyFrom, sum::MatrixSum},
};

mod add;
mod mul;

pub(crate) fn gradient_sum_over_axis<
    T: Num,
    M: ViewMemory<Item = T>,
    VM: ViewMutMemory<Item = T>,
>(
    source: Matrix<M, DimDyn>,
    target: Matrix<VM, DimDyn>,
) {
    if source.shape().len() < target.shape().len() {
        panic!("source.shape().len() < target.shape().len()");
    }

    let diff_len = source.shape().len() - target.shape().len();
    if diff_len == 0 {
        return;
    }

    if !source.shape().is_include(&target.shape()) {
        panic!("!source.shape().is_include(target.shape())");
    }

    if diff_len == 1 {
        let mut target = target;
        let ans = source.to_view().sum(0);
        target.to_view_mut().copy_from(&ans.to_view());
    } else {
        gradient_sum_over_axis(source.to_view().sum(0).to_view(), target);
    }
}

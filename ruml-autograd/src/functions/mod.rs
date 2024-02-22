use ruml_matrix::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::{Owned, View, ViewMut},
    num::Num,
    operation::{copy_from::CopyFrom, sum::MatrixSum},
};

use crate::Variable;

mod add;
mod mul;

pub mod matmul;

pub(crate) fn gradient_sum_over_axis<T: Num, M: View<Item = T>, VM: ViewMut<Item = T>>(
    source: Matrix<M, DimDyn>,
    target: Matrix<VM, DimDyn>,
) {
    if source.shape().len() < target.shape().len() {
        panic!("source.shape().len() < target.shape().len()");
    }

    let diff_len = source.shape().len() - target.shape().len();
    if diff_len == 0 {
        let mut target = target;
        target.to_view_mut().copy_from(&source.to_view());
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

pub(crate) fn output_shape<M: Owned>(x: &Variable<M>, y: &Variable<M>) -> DimDyn {
    if x.get_data().shape().is_include(&y.get_data().shape()) {
        x.get_data().shape()
    } else {
        y.get_data().shape()
    }
}

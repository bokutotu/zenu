use crate::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
};

use super::{
    asum::Asum,
    exp::{Exp, ExpAssign},
    max::MaxIdx,
};

pub trait SoftMax<T: Num> {
    fn softmax(&mut self, source: Matrix<ViewMem<T>, DimDyn>, axis: usize);
}

impl<'a, T: Num> SoftMax<T> for Matrix<ViewMutMem<'a, T>, DimDyn> {
    fn softmax(&mut self, source: Matrix<ViewMem<T>, DimDyn>, axis: usize) {
        if axis >= self.shape().len() {
            panic!("axis must be less than the number of dimensions");
        }
        self.to_view_mut().exp_assign(source);
    }
}

fn softmax_kernel_cpu<T: Num>(
    result: Matrix<ViewMutMem<T>, DimDyn>,
    source: Matrix<ViewMem<T>, DimDyn>,
) {
    let max_diff = source.clone() - source.clone().max();
    let exp = max_diff.exp();
    let sum = exp.asum();
    // result.to_view_mut().copy_from(&exp / sum);
}

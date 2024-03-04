use crate::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
};

use super::exp::Exp;

pub trait SoftMax<T: Num> {
    fn softmax(&mut self, source: Matrix<ViewMem<T>, DimDyn>, axis: usize);
}

impl<'a, T: Num> SoftMax<T> for Matrix<ViewMutMem<'a, T>, DimDyn> {
    fn softmax(&mut self, source: Matrix<ViewMem<T>, DimDyn>, axis: usize) {
        if axis >= self.shape().len() {
            panic!("axis must be less than the number of dimensions");
        }
        self.to_view_mut().exp(source);
    }
}

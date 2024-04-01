use crate::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, OwnedMatrix, ToViewMatrix},
    matrix_impl::{Matrix, OwnedMatrixDyn},
    memory::{Memory, ToViewMemory},
    memory_impl::OwnedMem,
    num::Num,
};

use super::{asum::Asum, sum::MatrixSum};

pub trait Mean<T: Num> {
    fn mean(&self, axis: Option<usize>, keep_dim: bool) -> Matrix<OwnedMem<T>, DimDyn>;
}

impl<T: Num, M: Memory<Item = T> + ToViewMemory, D: DimTrait> Mean<T> for Matrix<M, D> {
    fn mean(&self, axis: Option<usize>, keep_dim: bool) -> Matrix<OwnedMem<T>, DimDyn> {
        match axis {
            Some(axis) => {
                let sum_axis_num_elm = self.shape()[axis];

                let sum = self.to_view().into_dyn_dim().sum(axis, keep_dim);
                sum / T::from_usize(sum_axis_num_elm)
            }
            None => {
                let asum = self.to_view().asum();
                let num_elm = self.shape().num_elm();
                let mean = asum / T::from_usize(num_elm);
                OwnedMatrixDyn::from_vec(vec![mean], &[])
            }
        }
    }
}

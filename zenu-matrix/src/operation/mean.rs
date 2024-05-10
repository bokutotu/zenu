use crate::{
    device::Device,
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Repr},
    num::Num,
};

impl<T: Num, R: Repr<Item = T>, S: DimTrait, D: Device> Matrix<R, S, D> {
    pub fn mean(&self, axis: Option<usize>, keep_dim: bool) -> Matrix<Owned<T>, DimDyn, D> {
        match axis {
            Some(axis) => {
                let sum_axis_num_elm = self.shape()[axis];

                let sum = self.to_ref().into_dyn_dim().sum(axis, keep_dim);
                sum / T::from_usize(sum_axis_num_elm)
            }
            None => {
                let asum = self.to_ref().asum();
                let num_elm = self.shape().num_elm();
                let mean = asum / T::from_usize(num_elm);
                Matrix::from_vec(vec![mean], [])
            }
        }
    }
}

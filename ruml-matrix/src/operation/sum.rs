use crate::{
    dim::{DimDyn, DimTrait},
    index::index_dyn_impl::Index,
    matrix::{IndexAxisDyn, MatrixBase, OwnedMatrix, ToViewMutMatrix, ViewMatrix},
    matrix_impl::Matrix,
    memory::ViewMemory,
    operation::zeros::Zeros,
};

use super::add::MatrixAddAssign;

pub trait MatrixSum: ViewMatrix {
    type Output: OwnedMatrix;
    fn sum(self, axis: usize) -> Self::Output;
}

impl<M: ViewMemory, D: DimTrait> MatrixSum for Matrix<M, D> {
    type Output = Matrix<M::Owned, DimDyn>;
    fn sum(self, axis: usize) -> Self::Output {
        let self_dyn = self.into_dyn_dim();
        let shape = self_dyn.shape();
        if axis >= shape.len() {
            panic!("Invalid axis");
        }
        let result_shape = {
            let mut shape_ = DimDyn::default();
            for (i, &s) in shape.slice().iter().enumerate() {
                if i != axis {
                    shape_.push_dim(s);
                }
            }
            shape_
        };

        let mut result = Self::Output::zeros(result_shape);

        for i in 0..shape[axis] {
            let result_view_mut = result.to_view_mut();
            let s = self_dyn.clone();
            let s = s.index_axis_dyn(Index::new(axis, i));
            result_view_mut.add_assign(s);
        }

        result
    }
}

use crate::{
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{IndexAxisMutDyn, MatrixBase},
    matrix_impl::Matrix,
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
};

use super::copy_from::CopyFrom;

pub trait Broadcast<T: Num> {
    fn broadcast(&mut self, source: &Matrix<ViewMem<T>, DimDyn>);
}

impl<'a, T: Num> Broadcast<T> for Matrix<ViewMutMem<'a, T>, DimDyn> {
    fn broadcast(&mut self, source: &Matrix<ViewMem<T>, DimDyn>) {
        if !self.shape().is_include(source.shape()) {
            panic!("!self.shape().is_include(source.shape())");
        }

        if self.shape() == source.shape() {
            self.copy_from(source);
            return;
        }

        let diff_len = self.shape().len() - source.shape().len();

        if diff_len == 1 {
            for i in 0..self.shape()[0] {
                let mut to = self.index_axis_mut_dyn(Index0D::new(i));
                to.copy_from(source);
            }
        } else {
            for i in 0..self.shape()[0] {
                let mut to = self.index_axis_mut_dyn(Index0D::new(i));
                to.broadcast(source);
            }
        }
    }
}

#[cfg(test)]
mod broadcast {
    use crate::{
        dim::DimDyn,
        matrix::{OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::Matrix,
        memory_impl::OwnedMem,
        operation::{asum::Asum, zeros::Zeros},
    };

    use super::Broadcast;

    #[test]
    fn broadcast_2d_1d() {
        let source: Matrix<OwnedMem<f32>, DimDyn> = OwnedMatrix::from_vec(vec![1., 2., 3.], &[3]);
        let mut res: Matrix<OwnedMem<f32>, DimDyn> = Zeros::zeros([2, 3]);
        res.to_view_mut().broadcast(&source.to_view());
        let ans: Matrix<OwnedMem<f32>, DimDyn> =
            OwnedMatrix::from_vec(vec![1., 2., 3., 1., 2., 3.], &[2, 3]);
        let diff = ans.to_view() - res.to_view();
        let diff_sum = diff.to_view().asum();
        assert_eq!(diff_sum, 0.);
    }

    #[test]
    fn broadcast_4d_2d() {
        let source: Matrix<OwnedMem<f32>, DimDyn> = OwnedMatrix::from_vec(vec![1., 2.], &[1, 2]);
        let mut res: Matrix<OwnedMem<f32>, DimDyn> = Zeros::zeros([2, 3, 1, 2]);
        res.to_view_mut().broadcast(&source.to_view());
        let ans: Matrix<OwnedMem<f32>, DimDyn> = OwnedMatrix::from_vec(
            vec![1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.],
            &[2, 3, 1, 2],
        );
        let diff = ans.to_view() - res.to_view();
        let diff_sum = diff.to_view().asum();
        assert_eq!(diff_sum, 0.);
    }
}

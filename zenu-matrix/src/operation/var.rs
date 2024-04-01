use crate::{
    dim::DimDyn, matrix::ToViewMatrix, matrix_impl::Matrix, memory_impl::OwnedMem, num::Num,
};

use super::mean::Mean;

pub trait Variance<T: Num> {
    fn variance(&self, axis: Option<usize>, keep_dim: bool) -> Matrix<OwnedMem<T>, DimDyn>;
}

impl<T: Num> Variance<T> for Matrix<OwnedMem<T>, DimDyn> {
    fn variance(&self, axis: Option<usize>, keep_dim: bool) -> Matrix<OwnedMem<T>, DimDyn> {
        let mean = self.mean(axis, true);
        let diff = self.to_view() - mean;
        let diff = diff.to_view() * diff.to_view();
        diff.mean(axis, keep_dim)
    }
}

#[cfg(test)]
mod variance {
    use crate::{
        matrix::OwnedMatrix,
        matrix_impl::OwnedMatrixDyn,
        operation::{asum::Asum, var::Variance},
    };

    #[test]
    fn variance_1d() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let x = OwnedMatrixDyn::from_vec(x, &[4]);
        let ans = x.variance(None, false);
        assert!((ans - 1.25).asum() < 1e-6);
    }
    #[test]
    fn variance_1d_() {
        let x = vec![1.0, 2.0];
        let x = OwnedMatrixDyn::from_vec(x, &[2]);
        let ans = x.variance(None, false);
        assert!((ans - 0.25).asum() < 1e-6);
    }

    #[test]
    fn variance_2d() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let x = OwnedMatrixDyn::from_vec(x, &[2, 2]);
        let ans = x.variance(None, false);
        assert!((ans - 1.25).asum() < 1e-6);
    }

    #[test]
    fn variance_2d_0() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let x = OwnedMatrixDyn::from_vec(x, [2, 2]);
        let ans = x.variance(Some(0), false);
        assert!((ans - 1.0).asum() < 1e-6);
    }

    #[test]
    fn variance_2d_1() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let x = OwnedMatrixDyn::from_vec(x, [2, 2]);
        let ans = x.variance(Some(1), false);
        assert!((ans - 0.25).asum() < 1e-6);
    }
}

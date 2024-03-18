//! # Log Trait and Implementations
//!
//! This module provides the `Log` trait and its implementations for performing element-wise logarithm operations on matrices.
//!
//! ## Log Trait
//!
//! The `Log` trait defines two methods:
//!
//! - `log(&mut self, source: Matrix<ViewMem<Self::Item>, Self::Dim>)`: Computes the element-wise logarithm of the `source` matrix and stores the result in `self`.
//! - `log_assign(&mut self)`: Computes the element-wise logarithm of `self` in-place.
//!
//! ## Implementations
//!
//! The `Log` trait is implemented for `Matrix<SM, D>` where:
//! - `T` is the element type and must implement the `Num` trait.
//! - `D` is the dimension type and must implement the `DimTrait`.
//! - `SM` is the storage type and must implement the `ToViewMutMemory` trait with `Item = T`.
//!
//! The implementations handle both 1-dimensional and multi-dimensional matrices.
//!
//! ### 1-Dimensional Matrices
//!
//! For 1-dimensional matrices, the logarithm is computed using the `log_1d_cpu` function, which performs the operation using CPU.
//!
//! ### Multi-Dimensional Matrices
//!
//! For multi-dimensional matrices, the logarithm is computed by iterating over the first dimension and recursively calling the `log` or `log_assign` method on the sub-matrices.
//!
//! ## Functions
//!
//! - `log_1d_cpu<T, DM, SM, D1, D2>(dest: Matrix<DM, D1>, source: Matrix<SM, D2>)`: Computes the element-wise logarithm of `source` and stores the result in `dest` for 1-dimensional matrices using CPU.
//! - `log_1d_cpu_assign<T, M, D>(dest: Matrix<M, D>)`: Computes the element-wise logarithm of `dest` in-place for 1-dimensional matrices using CPU.
//!
//! ## Testing
//!
//! The module includes unit tests to verify the correctness of the `Log` trait implementations. The tests cover different scenarios, such as 1-dimensional and 2-dimensional matrices, and compare the results against ex//!
//! ## Example
//!
//! ```rust
//! use zenu_matrix::{matrix::{OwnedMatrix, ToViewMatrix}, matrix_impl::OwnedMatrixDyn, operation::log::Log};
//!
//! let mut a = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0], [3]);
//! a.log_assign();
//!
//! let mut b = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
//! let c = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
//! b.log(c.to_view());
//! ```

use crate::{
    dim::{Dim1, DimTrait},
    index::Index0D,
    matrix::{IndexAxisDyn, IndexAxisMutDyn, MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::{matrix_into_dim, Matrix},
    memory::{ToViewMutMemory, View, ViewMut},
    memory_impl::ViewMem,
    num::Num,
};

/// Trait for performing element-wise logarithm operations on matrices.
pub trait Log: ToViewMutMatrix + MatrixBase {
    /// Computes the element-wise logarithm of `source` and stores the result in `self`.
    ///
    /// # Panics
    ///
    /// Panics if the shapes of `self` and `source` do not match.
    fn log(&mut self, source: Matrix<ViewMem<Self::Item>, Self::Dim>);

    /// Computes the element-wise logarithm of `self` in-place.
    fn log_assign(&mut self);
}

impl<T, D, SM> Log for Matrix<SM, D>
where
    T: Num,
    D: DimTrait,
    SM: ToViewMutMemory<Item = T>,
{
    fn log(&mut self, source: Matrix<ViewMem<T>, D>) {
        if self.shape().slice() != source.shape().slice() {
            panic!("shape mismatch");
        }

        if self.shape().len() == 1 {
            log_1d_cpu(self.to_view_mut(), source.to_view());
        } else {
            for i in 0..self.shape()[0] {
                let mut dest = self.index_axis_mut_dyn(Index0D::new(i));
                let source = source.index_axis_dyn(Index0D::new(i));
                dest.log(source);
            }
        }
    }

    fn log_assign(&mut self) {
        if self.shape().len() == 1 {
            log_1d_cpu_assign(self.to_view_mut());
        } else {
            for i in 0..self.shape()[0] {
                let mut dest = self.index_axis_mut_dyn(Index0D::new(i));
                Log::log_assign(&mut dest);
            }
        }
    }
}

/// Computes the element-wise logarithm of `source` and stores the result in `dest` for 1-dimensional matrices using CPU.
fn log_1d_cpu<T, DM, SM, D1, D2>(dest: Matrix<DM, D1>, source: Matrix<SM, D2>)
where
    T: Num,
    DM: ViewMut<Item = T>,
    SM: View<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
{
    let source_stride = dest.stride()[0];
    let dest_stride = source.stride()[0];
    let mut dest: Matrix<DM, Dim1> = matrix_into_dim(dest);
    let source: Matrix<SM, Dim1> = matrix_into_dim(source);
    let dest = dest.as_mut_slice();
    let source = source.as_slice();
    for i in 0..source.len() {
        dest[i * dest_stride] = source[i * source_stride].ln();
    }
}

/// Computes the element-wise logarithm of `dest` in-place for 1-dimensional matrices using CPU.
fn log_1d_cpu_assign<T, M, D>(dest: Matrix<M, D>)
where
    T: Num,
    M: ViewMut<Item = T>,
    D: DimTrait,
{
    let dest_stride = dest.stride()[0];
    let mut dest: Matrix<M, Dim1> = matrix_into_dim(dest);
    let dest = dest.as_mut_slice();
    for i in 0..dest.len() {
        dest[i * dest_stride] = dest[i * dest_stride].ln();
    }
}

#[cfg(test)]
mod log {
    use crate::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::asum::Asum,
    };

    use super::Log;

    #[test]
    fn log_1d_assign() {
        let mut a = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0], [3]);
        a.log_assign();
        let ans = OwnedMatrixDyn::from_vec(vec![0.0, 0.6931471805599453, 1.0986122886681098], [3]);
        let diff = a - ans;
        let diff_asum = diff.asum();
        assert!(diff_asum < 1.0e-10);
    }

    #[test]
    fn log_1d() {
        let mut a = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0], [3]);
        a.log(b.to_view());
        let ans = OwnedMatrixDyn::from_vec(vec![0.0, 0.6931471805599453, 1.0986122886681098], [3]);
        let diff = a - ans;
        let diff_asum = diff.asum();
        assert!(diff_asum < 1.0e-10);
    }

    #[test]
    fn log_2d() {
        let mut a = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let b = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        a.log(b.to_view());
        let ans = OwnedMatrixDyn::from_vec(
            vec![
                0.0,
                0.6931471805599453,
                1.0986122886681098,
                1.3862943611198906,
                1.6094379124341003,
                1.791759469228055,
            ],
            [2, 3],
        );
        let diff = a - ans;
        let diff_asum = diff.asum();
        assert!(diff_asum < 1.0e-10);
    }

    #[test]
    fn log_2d_assign() {
        let mut a = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        a.log_assign();
        let ans = OwnedMatrixDyn::from_vec(
            vec![
                0.0,
                0.6931471805599453,
                1.0986122886681098,
                1.3862943611198906,
                1.6094379124341003,
                1.791759469228055,
            ],
            [2, 3],
        );
        let diff = a - ans;
        let diff_asum = diff.asum();
        assert!(diff_asum < 1.0e-10);
    }
}

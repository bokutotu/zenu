use crate::{
    dim::DimTrait,
    matrix::{IndexItemAsign, MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_blas::{
        dot::{dot_batch_shape_check, dot_batch_unchecked, dot_shape_check, dot_unchecked},
        gemm::{gemm_shape_check, gemm_unchecked},
    },
    matrix_impl::{matrix_into_dim, Matrix},
    memory::{ToViewMemory, ViewMut},
    num::Num,
};

/// Trait for computing the dot product of two vectors or matrices.
/// The behavior of the `dot` function depends on the dimensions of the input:
/// - If both inputs are 1-D arrays, it computes the inner product of vectors (without complex conjugation).
/// - If both inputs are 2-D arrays, it performs matrix multiplication. For this case, consider using `matmul`.
/// - If either input is 0-D (scalar), it multiplies the scalar with the other input. For this case, consider using `multiply` or the `*` operator.
/// - If the first input is an N-D array and the second input is a 1-D array, it computes a sum product over the last axis of the first input.
/// - If the first input is an N-D array and the second input is an M-D array (where M>=2), it computes a sum product over the last axis of the first input and the second-to-last axis of the second input.
///
/// # Examples
///
/// ```
/// use zenu_matrix::{
///     matrix::{IndexItem, OwnedMatrix, ToViewMutMatrix},
///     matrix_impl::{OwnedMatrix1D, OwnedMatrix2D},
/// };
///
/// use zenu_matrix::operation::dot::Dot;
///
/// let a = OwnedMatrix2D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
/// let b = OwnedMatrix1D::from_vec(vec![7.0, 8.0, 9.0], [3]);
/// let mut c = OwnedMatrix1D::from_vec(vec![0.0, 0.0], [2]);
/// c.to_view_mut().dot(a, b);
///
/// assert_eq!(c.index_item([0]), 50.0);
/// assert_eq!(c.index_item([1]), 122.0);
/// ```
pub trait Dot<RHS, LHS> {
    fn dot(self, rhs: RHS, lhs: LHS);
}

impl<T, SM, RM, LM, DS, DR, DL> Dot<Matrix<RM, DR>, Matrix<LM, DL>> for Matrix<SM, DS>
where
    T: Num,
    SM: ViewMut<Item = T>,
    RM: ToViewMemory<Item = T>,
    LM: ToViewMemory<Item = T>,
    DS: DimTrait,
    DR: DimTrait,
    DL: DimTrait,
{
    fn dot(self, rhs: Matrix<RM, DR>, lhs: Matrix<LM, DL>) {
        // if let Ok(())にするように書き換えて
        if let Ok(()) = dot_shape_check(rhs.shape(), lhs.shape()) {
            let result = dot_unchecked(
                matrix_into_dim(rhs).to_view(),
                matrix_into_dim(lhs).to_view(),
            );
            self.into_dyn_dim().index_item_asign([], result);
            return;
        }
        if let Ok(()) = dot_batch_shape_check(self.shape(), rhs.shape(), lhs.shape()) {
            dot_batch_unchecked(self, rhs, lhs);
            return;
        }
        if let Ok(()) = gemm_shape_check(&rhs, &lhs, &self) {
            gemm_unchecked(
                matrix_into_dim(rhs).to_view(),
                matrix_into_dim(lhs).to_view(),
                matrix_into_dim(self).to_view_mut(),
                T::one(),
                T::one(),
            );
            return;
        }
        panic!("Dimension mismatch");
    }
}

#[cfg(test)]
mod dot {
    use crate::{
        matrix::{IndexItem, OwnedMatrix, ToViewMutMatrix},
        matrix_impl::{OwnedMatrix0D, OwnedMatrix1D, OwnedMatrix2D},
    };

    use super::Dot;

    #[test]
    fn dot() {
        let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b = OwnedMatrix1D::from_vec(vec![4.0, 5.0, 6.0], [3]);
        let mut c = OwnedMatrix0D::from_vec(vec![0.0], []);
        c.to_view_mut().dot(a, b);

        assert_eq!(c.index_item([]), 32.0);
    }

    #[test]
    fn dot_2d_1d() {
        let a = OwnedMatrix2D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let b = OwnedMatrix1D::from_vec(vec![7.0, 8.0, 9.0], [3]);
        let mut c = OwnedMatrix1D::from_vec(vec![0.0, 0.0], [2]);
        c.to_view_mut().dot(a, b);

        assert_eq!(c.index_item([0]), 50.0);
        assert_eq!(c.index_item([1]), 122.0);
    }

    #[test]
    fn dot_1d_2d() {
        let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b = OwnedMatrix2D::from_vec(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [2, 3]);
        let mut c = OwnedMatrix1D::from_vec(vec![0.0, 0.0], [2]);
        c.to_view_mut().dot(a, b);

        assert_eq!(c.index_item([0]), 32.0);
        assert_eq!(c.index_item([1]), 50.0);
    }

    #[test]
    fn dot_2d_2d() {
        let a = OwnedMatrix2D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let b = OwnedMatrix2D::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2]);
        let mut c = OwnedMatrix2D::from_vec(vec![0.0, 0.0, 0.0, 0.0], [2, 2]);
        c.to_view_mut().dot(a, b);

        assert_eq!(c.index_item([0, 0]), 58.0);
        assert_eq!(c.index_item([0, 1]), 64.0);
        assert_eq!(c.index_item([1, 0]), 139.0);
        assert_eq!(c.index_item([1, 1]), 154.0);
    }
}

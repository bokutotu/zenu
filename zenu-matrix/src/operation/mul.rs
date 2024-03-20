use crate::{
    dim::DimTrait,
    matrix::ToViewMutMatrix,
    matrix_blas::gemm::{
        gemm_batch_shape_check, gemm_batch_unchecked, gemm_shape_check, gemm_unchecked,
    },
    matrix_impl::{matrix_into_dim, Matrix},
    memory::{ToViewMemory, View, ViewMut},
    num::Num,
};

/// Trait for computing the General Matrix Multiply (GEMM) operation.
///
/// The `gemm` function performs a matrix multiplication operation.
/// It takes two matrices as input and multiplies them together, storing the result in `self`.
///
/// # Shape Requirements
///
/// - `self`: The output matrix, must be a 2-D matrix.
/// - `rhs`: The right-hand side input matrix, must be a 2-D matrix.
/// - `lhs`: The left-hand side input matrix, must be a 2-D matrix.
///
/// The shapes of the input matrices must satisfy the following conditions:
/// - The number of columns of `rhs` must match the number of rows of `lhs`.
/// - The number of rows of `self` must match the number of rows of `rhs`.
/// - The number of columns of `self` must match the number of columns of `lhs`.
///
/// If the input matrices are higher-dimensional (3-D or more), the leading dimensions are
/// treated as batch dimensions, and the last two dimensions are used for matrix multiplication.
///
/// # Panics
///
/// This function will panic if:
/// - The shapes of the input matrices do not satisfy the above conditions.
/// - The dimensions of the input and output matrices are not greater than zero.
///
/// # Examples
///
/// ```
/// use zenu_matrix::{
///     matrix::{IndexItem, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
///     matrix_impl::OwnedMatrix2D,
///     constructor::zeros::Zeros,
/// };
///
/// use zenu_matrix::operation::mul::Gemm;
///
/// let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
/// let b = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], [3, 4]);
/// let mut ans = OwnedMatrix2D::<f32>::zeros([2, 4]);
///
/// ans.to_view_mut().gemm(a.to_view(), b.to_view());
///
/// assert_eq!(ans.index_item([0, 0]), 38.);
/// assert_eq!(ans.index_item([0, 1]), 44.);
/// assert_eq!(ans.index_item([0, 2]), 50.);
/// assert_eq!(ans.index_item([0, 3]), 56.);
/// assert_eq!(ans.index_item([1, 0]), 83.);
/// assert_eq!(ans.index_item([1, 1]), 98.);
/// assert_eq!(ans.index_item([1, 2]), 113.);
/// assert_eq!(ans.index_item([1, 3]), 128.);
/// ```
pub trait Gemm<Rhs, Lhs>: ToViewMutMatrix {
    /// Performs the General Matrix Multiply (GEMM) operation.
    ///
    /// This function takes two matrices as input and multiplies them together, storing the result in `self`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side matrix.
    /// * `lhs` - The left-hand side matrix.
    ///
    /// # Panics
    ///
    /// This function will panic if the dimensions of the matrices do not allow for matrix multiplication.
    fn gemm(self, rhs: Rhs, lhs: Lhs);
}

impl<T, M1, M2, M3, D1, D2, D3> Gemm<Matrix<M1, D1>, Matrix<M2, D2>> for Matrix<M3, D3>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    D3: DimTrait,
    M1: ToViewMemory<Item = T> + View,
    M2: ToViewMemory<Item = T> + View,
    M3: ViewMut<Item = T>,
{
    fn gemm(self, rhs: Matrix<M1, D1>, lhs: Matrix<M2, D2>) {
        // したのコードをif let Ok(())に続く形で書き直して
        if let Ok(()) = gemm_shape_check(&rhs, &lhs, &self) {
            gemm_unchecked(
                matrix_into_dim(rhs),
                matrix_into_dim(lhs),
                matrix_into_dim(self),
                T::one(),
                T::zero(),
            );
            return;
        }
        if let Ok(()) = gemm_batch_shape_check(&rhs, &lhs, &self) {
            gemm_batch_unchecked(rhs, lhs, self, T::one(), T::zero());
            return;
        }

        panic!("Dimension mismatch");
    }
}

#[cfg(test)]
mod mat_mul {
    use crate::{
        constructor::zeros::Zeros,
        matrix::{IndexItem, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::{OwnedMatrix2D, OwnedMatrix3D},
        operation::transpose::Transpose,
    };

    use super::*;

    #[test]
    fn default() {
        let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let b = OwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
            ],
            [3, 5],
        );
        let mut ans = OwnedMatrix2D::<f32>::zeros([2, 5]);

        ans.to_view_mut().gemm(a.to_view(), b.to_view());
        assert_eq!(ans.index_item([0, 0]), 46.);
        assert_eq!(ans.index_item([0, 1]), 52.);
        assert_eq!(ans.index_item([0, 2]), 58.);
        assert_eq!(ans.index_item([0, 3]), 64.);
        assert_eq!(ans.index_item([0, 4]), 70.);
        assert_eq!(ans.index_item([1, 0]), 100.);
        assert_eq!(ans.index_item([1, 1]), 115.);
        assert_eq!(ans.index_item([1, 2]), 130.);
        assert_eq!(ans.index_item([1, 3]), 145.);
        assert_eq!(ans.index_item([1, 4]), 160.);
    }

    #[test]
    fn default_stride_2() {
        let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        // shape 3 4
        let b = OwnedMatrix2D::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [3, 4],
        );
        let mut ans = OwnedMatrix2D::<f32>::zeros([2, 4]);

        ans.to_view_mut().gemm(a.to_view(), b.to_view());

        assert_eq!(ans.index_item([0, 0]), 38.);
        assert_eq!(ans.index_item([0, 1]), 44.);
        assert_eq!(ans.index_item([0, 2]), 50.);
        assert_eq!(ans.index_item([0, 3]), 56.);
        assert_eq!(ans.index_item([1, 0]), 83.);
        assert_eq!(ans.index_item([1, 1]), 98.);
        assert_eq!(ans.index_item([1, 2]), 113.);
        assert_eq!(ans.index_item([1, 3]), 128.);
    }

    #[test]
    fn gemm_2d() {
        let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], [2, 2]);
        let b = OwnedMatrix2D::from_vec(vec![5., 6., 7., 8.], [2, 2]);
        let mut c = OwnedMatrix2D::<f32>::zeros([2, 2]);

        c.to_view_mut().gemm(a.to_view(), b.to_view());

        assert_eq!(c.index_item([0, 0]), 19.);
        assert_eq!(c.index_item([0, 1]), 22.);
        assert_eq!(c.index_item([1, 0]), 43.);
        assert_eq!(c.index_item([1, 1]), 50.);
    }

    #[test]
    fn gemm_3d() {
        let a = OwnedMatrix3D::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [2, 2, 3],
        );
        let b = OwnedMatrix3D::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [2, 3, 2],
        );
        let mut c = OwnedMatrix3D::<f32>::zeros([2, 2, 2]);

        c.to_view_mut().gemm(a.to_view(), b.to_view());

        assert_eq!(c.index_item([0, 0, 0]), 22.);
        assert_eq!(c.index_item([0, 0, 1]), 28.);
        assert_eq!(c.index_item([0, 1, 0]), 49.);
        assert_eq!(c.index_item([0, 1, 1]), 64.);
        assert_eq!(c.index_item([1, 0, 0]), 220.);
        assert_eq!(c.index_item([1, 0, 1]), 244.);
        assert_eq!(c.index_item([1, 1, 0]), 301.);
        assert_eq!(c.index_item([1, 1, 1]), 334.);
    }

    #[test]
    fn gemm_transposed_a() {
        let mut a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], [2, 2]);
        let b = OwnedMatrix2D::from_vec(vec![5., 6., 7., 8.], [2, 2]);
        let mut c = OwnedMatrix2D::<f32>::zeros([2, 2]);

        a.transpose();
        c.to_view_mut().gemm(a.to_view(), b.to_view());

        assert_eq!(c.index_item([0, 0]), 26.);
        assert_eq!(c.index_item([0, 1]), 30.);
        assert_eq!(c.index_item([1, 0]), 38.);
        assert_eq!(c.index_item([1, 1]), 44.);
    }

    #[test]
    fn gemm_transposed_b() {
        let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], [2, 2]);
        let mut b = OwnedMatrix2D::from_vec(vec![5., 6., 7., 8.], [2, 2]);
        let mut c = OwnedMatrix2D::<f32>::zeros([2, 2]);

        b.transpose();
        c.to_view_mut().gemm(a.to_view(), b.to_view());

        assert_eq!(c.index_item([0, 0]), 17.);
        assert_eq!(c.index_item([0, 1]), 23.);
        assert_eq!(c.index_item([1, 0]), 39.);
        assert_eq!(c.index_item([1, 1]), 53.);
    }

    #[test]
    fn gemm_transposed_a_and_b() {
        let mut a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], [2, 2]);
        let mut b = OwnedMatrix2D::from_vec(vec![5., 6., 7., 8.], [2, 2]);
        let mut c = OwnedMatrix2D::<f32>::zeros([2, 2]);

        a.transpose();
        b.transpose();
        c.to_view_mut().gemm(a.to_view(), b.to_view());

        assert_eq!(c.index_item([0, 0]), 23.);
        assert_eq!(c.index_item([0, 1]), 31.);
        assert_eq!(c.index_item([1, 0]), 34.);
        assert_eq!(c.index_item([1, 1]), 46.);
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn gemm_dimension_mismatch() {
        let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], [2, 2]);
        let b = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]);
        let mut c = OwnedMatrix2D::<f32>::zeros([2, 2]);

        c.to_view_mut().gemm(a.to_view(), b.to_view());
    }
}

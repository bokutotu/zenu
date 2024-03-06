use crate::{
    dim::DimTrait,
    matrix::{MatrixBase, ViewMutMatix},
    matrix_blas::gemm::gemm,
    matrix_impl::{matrix_into_dim, Matrix},
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
};

pub trait Gemm<Rhs, Lhs>: ViewMutMatix {
    fn gemm(self, rhs: Rhs, lhs: Lhs);
}

impl<'a, 'b, 'c, T, D1, D2, D3> Gemm<Matrix<ViewMem<'a, T>, D1>, Matrix<ViewMem<'b, T>, D2>>
    for Matrix<ViewMutMem<'c, T>, D3>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    D3: DimTrait,
{
    fn gemm(self, rhs: Matrix<ViewMem<T>, D1>, lhs: Matrix<ViewMem<T>, D2>) {
        assert_eq!(self.shape().len(), 2);
        assert_eq!(rhs.shape().len(), 2);
        assert_eq!(lhs.shape().len(), 2);
        let self_ = matrix_into_dim(self);
        let rhs = matrix_into_dim(rhs);
        let lhs = matrix_into_dim(lhs);
        gemm(rhs, lhs, self_, T::one(), T::zero());
    }
}

#[cfg(test)]
mod mat_mul {
    use crate::{
        matrix::{IndexItem, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::OwnedMatrix2D,
        operation::zeros::Zeros,
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
        dbg!(ans.index_item([0, 0]));
        dbg!(ans.index_item([0, 1]));
        dbg!(ans.index_item([1, 0]));
        dbg!(ans.index_item([1, 1]));
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
        dbg!(ans.index_item([0, 0]));
        dbg!(ans.index_item([0, 1]));
        dbg!(ans.index_item([0, 2]));
        dbg!(ans.index_item([0, 3]));
        dbg!(ans.index_item([1, 0]));
        dbg!(ans.index_item([1, 1]));
        dbg!(ans.index_item([1, 2]));
        dbg!(ans.index_item([1, 3]));

        assert_eq!(ans.index_item([0, 0]), 38.);
        assert_eq!(ans.index_item([0, 1]), 44.);
        assert_eq!(ans.index_item([0, 2]), 50.);
        assert_eq!(ans.index_item([0, 3]), 56.);
        assert_eq!(ans.index_item([1, 0]), 83.);
        assert_eq!(ans.index_item([1, 1]), 98.);
        assert_eq!(ans.index_item([1, 2]), 113.);
        assert_eq!(ans.index_item([1, 3]), 128.);
    }
}

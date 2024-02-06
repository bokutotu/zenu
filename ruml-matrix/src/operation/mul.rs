use std::any::TypeId;

use crate::{
    dim,
    dim::DimTrait,
    dim_impl::{Dim0, Dim1, Dim2, Dim3, Dim4},
    index_impl::Index0D,
    matrix::{
        IndexAxis, IndexAxisMut, IndexItem, IndexItemAsign, MatrixBase, ViewMatrix, ViewMutMatix,
    },
    matrix_blas::gemm::gemm,
    matrix_impl::{matrix_into_dim, Matrix},
    memory::{ViewMemory, ViewMutMemory},
    num::Num,
};

fn mul_matrix_scalar<T, LM, SM, D>(self_: Matrix<SM, D>, lhs: Matrix<LM, D>, rhs: T)
where
    T: Num,
    SM: ViewMutMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    D: DimTrait,
{
    assert_eq!(self_.shape(), lhs.shape());

    macro_rules! mul_matrix_scalar_dim {
        ($self_:ident, $lhs:ident, $rhs:ident, $dim:ident) => {{
            let mut self_: Matrix<SM, $dim> = matrix_into_dim($self_);
            let lhs: Matrix<LM, $dim> = matrix_into_dim($lhs);
            let self_shape: $dim = self_.shape();
            for idx in 0..self_shape[0] {
                let self_row = self_.index_axis_mut(Index0D::new(idx));
                let lhs_row = lhs.index_axis(Index0D::new(idx));
                mul_matrix_scalar(self_row, lhs_row, $rhs);
            }
        }};
    }

    match self_.shape().len() {
        1 => {
            let mut self_: Matrix<SM, Dim1> = matrix_into_dim(self_);
            let lhs: Matrix<LM, Dim1> = matrix_into_dim(lhs);
            for idx in 0..self_.shape()[0] {
                self_.index_item_asign(dim!(idx), lhs.index_item(dim!(idx)) * rhs);
            }
        }
        2 => mul_matrix_scalar_dim!(self_, lhs, rhs, Dim2),
        3 => mul_matrix_scalar_dim!(self_, lhs, rhs, Dim3),
        4 => mul_matrix_scalar_dim!(self_, lhs, rhs, Dim4),
        _ => panic!("not implemented: this is bug. please report this bug."),
    }
}

// // TODO: クッソ汚いコードなのでどうにかする
fn mul_matrix_matrix<T, LM, RM, SM, D1, D2>(
    self_: Matrix<SM, D1>,
    lhs: Matrix<LM, D1>,
    rhs: Matrix<RM, D2>,
) where
    T: Num,
    LM: ViewMemory<Item = T>,
    RM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
{
    assert_eq!(self_.shape(), lhs.shape());
    assert!(self_.shape().len() >= rhs.shape().len());

    if TypeId::of::<D2>() == TypeId::of::<Dim0>() {
        let rhs: Matrix<RM, Dim0> = matrix_into_dim(rhs);
        let scalar = rhs.get_value();
        mul_matrix_scalar(self_, lhs, scalar);
    } else if TypeId::of::<D1>() == TypeId::of::<D2>() {
        macro_rules! impl_mul_same_dim {
            ($dim:ty) => {{
                let mut self_: Matrix<SM, $dim> = matrix_into_dim(self_);
                let lhs: Matrix<LM, $dim> = matrix_into_dim(lhs);
                let rhs: Matrix<RM, $dim> = matrix_into_dim(rhs);
                for idx in 0..self_.shape()[0] {
                    let self_ = self_.index_axis_mut(Index0D::new(idx));
                    let lhs = lhs.index_axis(Index0D::new(idx));
                    let rhs = rhs.index_axis(Index0D::new(idx));
                    mul_matrix_matrix(self_, lhs, rhs);
                }
            }};
        }
        match self_.shape().len() {
            1 => {
                let mut self_: Matrix<SM, Dim1> = matrix_into_dim(self_);
                let lhs: Matrix<LM, Dim1> = matrix_into_dim(lhs);
                let rhs: Matrix<RM, Dim1> = matrix_into_dim(rhs);
                for idx in 0..self_.shape()[0] {
                    self_.index_item_asign(
                        dim!(idx),
                        lhs.index_item(dim!(idx)) * rhs.index_item(dim!(idx)),
                    );
                }
            }
            2 => impl_mul_same_dim!(Dim2),
            3 => impl_mul_same_dim!(Dim3),
            4 => impl_mul_same_dim!(Dim4),
            _ => panic!("not implemented: this is bug. please report this bug."),
        }
    } else {
        macro_rules! impl_mul_diff_dim {
            ($dim1:ty, $dim2:ty) => {{
                let mut self_: Matrix<SM, $dim1> = matrix_into_dim(self_);
                let lhs: Matrix<LM, $dim1> = matrix_into_dim(lhs);
                let rhs: Matrix<RM, $dim2> = matrix_into_dim(rhs);
                for idx in 0..self_.shape()[0] {
                    let self_ = self_.index_axis_mut(Index0D::new(idx));
                    let lhs = lhs.index_axis(Index0D::new(idx));
                    mul_matrix_matrix(self_, lhs, rhs.clone());
                }
            }};
        }
        match (self_.shape().len(), rhs.shape().len()) {
            (2, 1) => impl_mul_diff_dim!(Dim2, Dim1),
            (3, 1) => impl_mul_diff_dim!(Dim3, Dim1),
            (3, 2) => impl_mul_diff_dim!(Dim3, Dim2),
            (4, 1) => impl_mul_diff_dim!(Dim4, Dim1),
            (4, 2) => impl_mul_diff_dim!(Dim4, Dim2),
            (4, 3) => impl_mul_diff_dim!(Dim4, Dim3),
            _ => panic!("not implemented: this is bug. please report this bug."),
        }
    }
}

pub trait MatrixMul<Lhs, Rhs>: ViewMutMatix {
    fn mul(self, lhs: Lhs, rhs: Rhs);
}

impl<T, RM, SM, D> MatrixMul<Matrix<RM, D>, T> for Matrix<SM, D>
where
    T: Num,
    RM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
    D: DimTrait,
{
    fn mul(self, lhs: Matrix<RM, D>, rhs: T) {
        mul_matrix_scalar(self, lhs, rhs);
    }
}

impl<T, DS, DR, SM, LM, RM> MatrixMul<Matrix<LM, DS>, Matrix<RM, DR>> for Matrix<SM, DS>
where
    T: Num,
    DS: DimTrait,
    DR: DimTrait,
    SM: ViewMutMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    RM: ViewMemory<Item = T>,
{
    fn mul(self, lhs: Matrix<LM, DS>, rhs: Matrix<RM, DR>) {
        mul_matrix_matrix(self, lhs, rhs);
    }
}

pub trait MatMul<Rhs, Lhs>: ViewMutMatix {
    fn mat_mul(self, rhs: Rhs, lhs: Lhs);
}

impl<T, S, R, L> MatMul<R, L> for S
where
    T: Num,
    L: ViewMatrix<Item = T> + MatrixBase<Dim = Dim2>,
    R: ViewMatrix<Item = T> + MatrixBase<Dim = Dim2>,
    S: ViewMutMatix<Item = T> + MatrixBase<Dim = Dim2>,
{
    fn mat_mul(self, rhs: R, lhs: L) {
        gemm(rhs, lhs, self, T::one(), T::zero());
    }
}

#[cfg(test)]
mod mul {
    use crate::{
        dim,
        matrix::{IndexItem, MatrixSlice, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::{CpuOwnedMatrix0D, CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix4D},
        operation::zeros::Zeros,
        slice,
    };

    use super::MatrixMul;

    #[test]
    fn mul_1d_scalar() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], dim!(3));
        let b = CpuOwnedMatrix0D::from_vec(vec![2.0], dim!());
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(3));
        ans.to_view_mut().mul(a.to_view(), b.to_view());

        assert_eq!(ans.index_item(dim!(0)), 2.0);
        assert_eq!(ans.index_item(dim!(1)), 4.0);
        assert_eq!(ans.index_item(dim!(2)), 6.0);
    }

    #[test]
    fn scalar_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3.], dim!(3));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(3));
        ans.to_view_mut().mul(a.to_view(), 2.);

        assert_eq!(ans.index_item(dim!(0)), 2.);
        assert_eq!(ans.index_item(dim!(1)), 4.);
        assert_eq!(ans.index_item(dim!(2)), 6.);
    }

    #[test]
    fn sliced_scalar_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], dim!(4));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(2));
        ans.to_view_mut().mul(a.to_view().slice(slice!(..;2)), 2.);

        assert_eq!(ans.index_item(dim!(0)), 2.);
        assert_eq!(ans.index_item(dim!(1)), 6.);
    }

    #[test]
    fn scalar_2d() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], dim![2, 3]);
        let mut ans = CpuOwnedMatrix2D::<f32>::zeros(dim![2, 3]);
        ans.to_view_mut().mul(a.to_view(), 2.);

        assert_eq!(ans.index_item(dim!(0, 0)), 2.);
        assert_eq!(ans.index_item(dim!(0, 1)), 4.);
        assert_eq!(ans.index_item(dim!(0, 2)), 6.);
        assert_eq!(ans.index_item(dim!(1, 0)), 8.);
        assert_eq!(ans.index_item(dim!(1, 1)), 10.);
        assert_eq!(ans.index_item(dim!(1, 2)), 12.);
    }

    #[test]
    fn default_1d_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3.], dim!(3));
        let b = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3.], dim!(3));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(3));
        ans.to_view_mut().mul(a.to_view(), b.to_view());

        assert_eq!(ans.index_item(dim!(0)), 1.);
        assert_eq!(ans.index_item(dim!(1)), 4.);
        assert_eq!(ans.index_item(dim!(2)), 9.);
    }

    #[test]
    fn sliced_1d_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], dim!(4));
        let b = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], dim!(4));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(2));
        ans.to_view_mut().mul(
            a.to_view().slice(slice!(..;2)),
            b.to_view().slice(slice!(..;2)),
        );

        assert_eq!(ans.index_item(dim!(0)), 1.);
        assert_eq!(ans.index_item(dim!(1)), 9.);
    }

    #[test]
    fn default_2d_2d() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], dim![2, 3]);
        let b = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], dim![2, 3]);
        let mut ans = CpuOwnedMatrix2D::<f32>::zeros(dim![2, 3]);
        ans.to_view_mut().mul(a.to_view(), b.to_view());

        assert_eq!(ans.index_item(dim!(0, 0)), 1.);
        assert_eq!(ans.index_item(dim!(0, 1)), 4.);
        assert_eq!(ans.index_item(dim!(0, 2)), 9.);
        assert_eq!(ans.index_item(dim!(1, 0)), 16.);
        assert_eq!(ans.index_item(dim!(1, 1)), 25.);
        assert_eq!(ans.index_item(dim!(1, 2)), 36.);
    }

    #[test]
    fn sliced_4d_2d() {
        let mut a_vec = Vec::new();
        for i in 0..2 * 2 * 2 * 2 {
            a_vec.push(i as f32);
        }

        let a = CpuOwnedMatrix4D::from_vec(a_vec, dim![2, 2, 2, 2]);
        let b = CpuOwnedMatrix1D::from_vec(vec![1., 2.], dim![2]);

        let mut ans = CpuOwnedMatrix4D::<f32>::zeros(dim![2, 2, 2, 2]);

        ans.to_view_mut().mul(a.to_view(), b.to_view());

        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        assert_eq!(
                            ans.index_item(dim!(i, j, k, l)),
                            a.index_item(dim!(i, j, k, l)) * b.index_item(dim!(l))
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod mat_mul {
    use crate::{
        dim,
        matrix::{IndexItem, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::CpuOwnedMatrix2D,
        operation::zeros::Zeros,
    };

    use super::*;

    #[test]
    fn default() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], dim![3, 2]);
        let b = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], dim![2, 3]);
        let mut ans = CpuOwnedMatrix2D::<f32>::zeros(dim![2, 2]);

        ans.to_view_mut().mat_mul(a.to_view(), b.to_view());
        dbg!(ans.index_item(dim!(0, 0)));
        dbg!(ans.index_item(dim!(0, 1)));
        dbg!(ans.index_item(dim!(1, 0)));
        dbg!(ans.index_item(dim!(1, 1)));
        assert_eq!(ans.index_item(dim!(0, 0)), 22.);
        assert_eq!(ans.index_item(dim!(0, 1)), 28.);
        assert_eq!(ans.index_item(dim!(1, 0)), 49.);
        assert_eq!(ans.index_item(dim!(1, 1)), 64.);
    }

    #[test]
    fn default_stride_2() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], dim![3, 2]);
        // shape 3 4
        let b = CpuOwnedMatrix2D::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            dim![4, 3],
        );
        let mut ans = CpuOwnedMatrix2D::<f32>::zeros(dim![4, 2]);

        ans.to_view_mut().mat_mul(a.to_view(), b.to_view());
        dbg!(ans.index_item(dim!(0, 0)));
        dbg!(ans.index_item(dim!(0, 1)));
        dbg!(ans.index_item(dim!(1, 0)));
        dbg!(ans.index_item(dim!(1, 1)));
        dbg!(ans.index_item(dim!(2, 0)));
        dbg!(ans.index_item(dim!(2, 1)));
        dbg!(ans.index_item(dim!(3, 0)));
        dbg!(ans.index_item(dim!(3, 1)));

        assert_eq!(ans.index_item(dim!(0, 0)), 38.);
        assert_eq!(ans.index_item(dim!(0, 1)), 44.);
        assert_eq!(ans.index_item(dim!(1, 0)), 50.);
        assert_eq!(ans.index_item(dim!(1, 1)), 56.);
        assert_eq!(ans.index_item(dim!(2, 0)), 83.);
        assert_eq!(ans.index_item(dim!(2, 1)), 98.);
        assert_eq!(ans.index_item(dim!(3, 0)), 113.);
        assert_eq!(ans.index_item(dim!(3, 1)), 128.);
    }
}

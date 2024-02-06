use std::any::TypeId;

use crate::{
    dim,
    dim::DimTrait,
    dim_impl::{Dim0, Dim1, Dim2, Dim3, Dim4},
    index_impl::Index0D,
    matrix::{IndexAxis, IndexAxisMut, IndexItem, IndexItemAsign, MatrixBase, ViewMutMatix},
    matrix_impl::{matrix_into_dim, Matrix},
    memory::{ViewMemory, ViewMutMemory},
    num::Num,
};

fn add_matrix_scalar<T, LM, SM, D>(self_: Matrix<SM, D>, lhs: Matrix<LM, D>, rhs: T)
where
    T: Num,
    SM: ViewMutMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    D: DimTrait,
{
    assert_eq!(self_.shape(), lhs.shape());

    macro_rules! add_matrix_scalar_dim {
        ($self_:ident, $lhs:ident, $rhs:ident, $dim:ident) => {{
            let mut self_: Matrix<SM, $dim> = matrix_into_dim($self_);
            let lhs: Matrix<LM, $dim> = matrix_into_dim($lhs);
            let self_shape: $dim = self_.shape();
            for idx in 0..self_shape[0] {
                let self_row = self_.index_axis_mut(Index0D::new(idx));
                let lhs_row = lhs.index_axis(Index0D::new(idx));
                add_matrix_scalar(self_row, lhs_row, $rhs);
            }
        }};
    }

    match self_.shape().len() {
        1 => {
            let mut self_: Matrix<SM, Dim1> = matrix_into_dim(self_);
            let lhs: Matrix<LM, Dim1> = matrix_into_dim(lhs);
            for idx in 0..self_.shape()[0] {
                self_.index_item_asign(dim!(idx), lhs.index_item(dim!(idx)) + rhs);
            }
        }
        2 => add_matrix_scalar_dim!(self_, lhs, rhs, Dim2),
        3 => add_matrix_scalar_dim!(self_, lhs, rhs, Dim3),
        4 => add_matrix_scalar_dim!(self_, lhs, rhs, Dim4),
        _ => panic!("not implemented: this is bug. please report this bug."),
    }
}

fn add_matrix_matrix<T, LM, RM, SM, D1, D2>(
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
        add_matrix_scalar(self_, lhs, scalar);
    } else if TypeId::of::<D1>() == TypeId::of::<D2>() {
        macro_rules! impl_add_same_dim {
            ($dim:ty) => {{
                let mut self_: Matrix<SM, $dim> = matrix_into_dim(self_);
                let lhs: Matrix<LM, $dim> = matrix_into_dim(lhs);
                let rhs: Matrix<RM, $dim> = matrix_into_dim(rhs);
                for idx in 0..self_.shape()[0] {
                    let self_ = self_.index_axis_mut(Index0D::new(idx));
                    let lhs = lhs.index_axis(Index0D::new(idx));
                    let rhs = rhs.index_axis(Index0D::new(idx));
                    add_matrix_matrix(self_, lhs, rhs);
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
                        lhs.index_item(dim!(idx)) + rhs.index_item(dim!(idx)),
                    );
                }
            }
            2 => impl_add_same_dim!(Dim2),
            3 => impl_add_same_dim!(Dim3),
            4 => impl_add_same_dim!(Dim4),
            _ => panic!("not implemented: this is bug. please report this bug."),
        }
    } else {
        macro_rules! impl_add_diff_dim {
            ($dim1:ty, $dim2:ty) => {{
                let mut self_: Matrix<SM, $dim1> = matrix_into_dim(self_);
                let lhs: Matrix<LM, $dim1> = matrix_into_dim(lhs);
                let rhs: Matrix<RM, $dim2> = matrix_into_dim(rhs);
                for idx in 0..self_.shape()[0] {
                    let self_ = self_.index_axis_mut(Index0D::new(idx));
                    let lhs = lhs.index_axis(Index0D::new(idx));
                    add_matrix_matrix(self_, lhs, rhs.clone());
                }
            }};
        }
        match (self_.shape().len(), rhs.shape().len()) {
            (2, 1) => impl_add_diff_dim!(Dim2, Dim1),
            (3, 1) => impl_add_diff_dim!(Dim3, Dim1),
            (3, 2) => impl_add_diff_dim!(Dim3, Dim2),
            (4, 1) => impl_add_diff_dim!(Dim4, Dim1),
            (4, 2) => impl_add_diff_dim!(Dim4, Dim2),
            (4, 3) => impl_add_diff_dim!(Dim4, Dim3),
            _ => panic!("not implemented: this is bug. please report this bug."),
        }
    }
}

pub trait MatrixAdd<Rhs, Lhs>: ViewMutMatix + MatrixBase {
    fn add(self, lhs: Rhs, rhs: Lhs);
}

// matrix add scalar
impl<T, RM, SM, D> MatrixAdd<Matrix<RM, D>, T> for Matrix<SM, D>
where
    T: Num,
    RM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
    D: DimTrait,
{
    fn add(self, lhs: Matrix<RM, D>, rhs: T) {
        add_matrix_scalar(self, lhs, rhs);
    }
}

impl<T, RM, LM, SM, D1, D2> MatrixAdd<Matrix<LM, D1>, Matrix<RM, D2>> for Matrix<SM, D1>
where
    T: Num,
    RM: ViewMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
{
    fn add(self, lhs: Matrix<LM, D1>, rhs: Matrix<RM, D2>) {
        add_matrix_matrix(self, lhs, rhs);
    }
}

#[cfg(test)]
mod add {
    use crate::{
        dim,
        matrix::{MatrixSlice, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::{CpuOwnedMatrix0D, CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix3D},
        operation::zeros::Zeros,
        slice,
    };

    use super::*;

    #[test]
    fn add_1d_scalar() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], dim!(3));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(3));
        let b = CpuOwnedMatrix0D::from_vec(vec![2.0], dim!());
        ans.to_view_mut().add(a.to_view(), b.to_view());

        assert_eq!(ans.index_item(dim!(0)), 3.0);
        assert_eq!(ans.index_item(dim!(1)), 4.0);
        assert_eq!(ans.index_item(dim!(2)), 5.0);
    }

    #[test]
    fn add_1d_scalar_default_stride() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], dim!(3));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(3));
        ans.to_view_mut().add(a.to_view(), 1.0);

        assert_eq!(ans.index_item(dim!(0)), 2.0);
        assert_eq!(ans.index_item(dim!(1)), 3.0);
        assert_eq!(ans.index_item(dim!(2)), 4.0);
    }

    #[test]
    fn add_1d_scalar_sliced() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dim!(6));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(3));

        let sliced = a.slice(slice!(..;2));

        ans.to_view_mut().add(sliced.to_view(), 1.0);

        assert_eq!(ans.index_item(dim!(0)), 2.0);
        assert_eq!(ans.index_item(dim!(1)), 4.0);
        assert_eq!(ans.index_item(dim!(2)), 6.0);
    }

    #[test]
    fn add_3d_scalar_sliced() {
        let a = CpuOwnedMatrix3D::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
            ],
            dim!(3, 3, 4),
        );

        let mut ans = CpuOwnedMatrix3D::<f32>::zeros(dim!(3, 3, 2));

        let sliced = a.slice(slice!(.., .., ..;2));

        ans.to_view_mut().add(sliced.to_view(), 1.0);

        assert_eq!(ans.index_item(dim!(0, 0, 0)), 2.0);
        assert_eq!(ans.index_item(dim!(0, 0, 1)), 4.0);
        assert_eq!(ans.index_item(dim!(0, 1, 0)), 6.0);
        assert_eq!(ans.index_item(dim!(0, 1, 1)), 8.0);
        assert_eq!(ans.index_item(dim!(0, 2, 0)), 10.0);
        assert_eq!(ans.index_item(dim!(0, 2, 1)), 12.0);
        assert_eq!(ans.index_item(dim!(1, 0, 0)), 14.0);
        assert_eq!(ans.index_item(dim!(1, 0, 1)), 16.0);
        assert_eq!(ans.index_item(dim!(1, 1, 0)), 18.0);
        assert_eq!(ans.index_item(dim!(1, 1, 1)), 20.0);
        assert_eq!(ans.index_item(dim!(1, 2, 0)), 22.0);
        assert_eq!(ans.index_item(dim!(1, 2, 1)), 24.0);
        assert_eq!(ans.index_item(dim!(2, 0, 0)), 26.0);
        assert_eq!(ans.index_item(dim!(2, 0, 1)), 28.0);
        assert_eq!(ans.index_item(dim!(2, 1, 0)), 30.0);
        assert_eq!(ans.index_item(dim!(2, 1, 1)), 32.0);
        assert_eq!(ans.index_item(dim!(2, 2, 0)), 34.0);
        assert_eq!(ans.index_item(dim!(2, 2, 1)), 36.0);
    }

    #[test]
    fn add_1d_1d_default_stride() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], dim!(3));
        let b = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], dim!(3));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(3));
        ans.to_view_mut().add(a.to_view(), b.to_view());

        assert_eq!(ans.index_item(dim!(0)), 2.0);
        assert_eq!(ans.index_item(dim!(1)), 4.0);
        assert_eq!(ans.index_item(dim!(2)), 6.0);
    }

    #[test]
    fn add_1d_1d_sliced() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dim!(6));
        let b = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dim!(6));

        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(3));

        let sliced_a = a.slice(slice!(..;2));
        let sliced_b = b.slice(slice!(1..;2));

        ans.to_view_mut()
            .add(sliced_a.to_view(), sliced_b.to_view());

        assert_eq!(ans.index_item(dim!(0)), 3.0);
        assert_eq!(ans.index_item(dim!(1)), 7.0);
        assert_eq!(ans.index_item(dim!(2)), 11.0);
    }

    #[test]
    fn add_2d_1d_default() {
        let a = CpuOwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            dim!(4, 4),
        );

        let b = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], dim!(8));

        let mut ans = CpuOwnedMatrix2D::<f32>::zeros(dim!(2, 2));

        let sliced_a = a.slice(slice!(..2, ..2));
        let sliced_b = b.slice(slice!(..2));

        ans.to_view_mut()
            .add(sliced_a.to_view(), sliced_b.to_view());

        assert_eq!(ans.index_item(dim!(0, 0)), 2.0);
        assert_eq!(ans.index_item(dim!(0, 1)), 4.0);
        assert_eq!(ans.index_item(dim!(1, 0)), 6.0);
        assert_eq!(ans.index_item(dim!(1, 1)), 8.0);
    }

    #[test]
    fn add_3d_1d_sliced() {
        let mut v = Vec::new();
        let num_elm = 4 * 4 * 4;
        for i in 0..num_elm {
            v.push(i as f32);
        }
        let a = CpuOwnedMatrix3D::from_vec(v, dim!(4, 4, 4));

        let b = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], dim!(4));

        let mut ans = CpuOwnedMatrix3D::<f32>::zeros(dim!(2, 2, 2));

        let sliced_a = a.slice(slice!(..2, 1..;2, ..2));
        let sliced_b = b.slice(slice!(..2));

        ans.to_view_mut()
            .add(sliced_a.to_view(), sliced_b.to_view());

        assert_eq!(ans.index_item(dim!(0, 0, 0)), 5.);
        assert_eq!(ans.index_item(dim!(0, 0, 1)), 7.);
        assert_eq!(ans.index_item(dim!(0, 1, 0)), 13.);
        assert_eq!(ans.index_item(dim!(0, 1, 1)), 15.);
        assert_eq!(ans.index_item(dim!(1, 0, 0)), 21.);
        assert_eq!(ans.index_item(dim!(1, 0, 1)), 23.);
        assert_eq!(ans.index_item(dim!(1, 1, 0)), 29.);
        assert_eq!(ans.index_item(dim!(1, 1, 1)), 31.);
    }

    #[test]
    fn add_2d_2d_default() {
        let a = CpuOwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            dim!(4, 4),
        );

        let b = CpuOwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            dim!(4, 4),
        );

        let mut ans = CpuOwnedMatrix2D::<f32>::zeros(dim!(4, 4));
        ans.to_view_mut().add(a.to_view(), b.to_view());

        assert_eq!(ans.index_item(dim!(0, 0)), 2.0);
        assert_eq!(ans.index_item(dim!(0, 1)), 4.0);
        assert_eq!(ans.index_item(dim!(0, 2)), 6.0);
        assert_eq!(ans.index_item(dim!(0, 3)), 8.0);
        assert_eq!(ans.index_item(dim!(1, 0)), 10.0);
        assert_eq!(ans.index_item(dim!(1, 1)), 12.0);
        assert_eq!(ans.index_item(dim!(1, 2)), 14.0);
        assert_eq!(ans.index_item(dim!(1, 3)), 16.0);
        assert_eq!(ans.index_item(dim!(2, 0)), 18.0);
        assert_eq!(ans.index_item(dim!(2, 1)), 20.0);
        assert_eq!(ans.index_item(dim!(2, 2)), 22.0);
        assert_eq!(ans.index_item(dim!(2, 3)), 24.0);
        assert_eq!(ans.index_item(dim!(3, 0)), 26.0);
        assert_eq!(ans.index_item(dim!(3, 1)), 28.0);
        assert_eq!(ans.index_item(dim!(3, 2)), 30.0);
        assert_eq!(ans.index_item(dim!(3, 3)), 32.0);
    }
}

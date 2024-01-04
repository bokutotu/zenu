use crate::{
    blas::Blas,
    dim,
    dim::DimTrait,
    dim_impl::{Dim1, Dim2, Dim3, Dim4},
    index_impl::Index0D,
    matrix::{
        AsMutPtr, AsPtr, IndexAxis, IndexAxisMut, IndexItem, IndexItemAsign, MatrixBase,
        ViewMutMatix,
    },
    matrix_impl::Matrix,
    memory::{Memory, ViewMemory, ViewMutMemory},
    num::Num,
    operation::copy_from::CopyFrom,
};

pub trait MatrixAdd<Rhs, Lhs, T>: ViewMutMatix + MatrixBase<Item = T> {
    fn add(&mut self, rhs: &Rhs, lhs: &Lhs);
}

// matrix add scalar
impl<T, RM, SM> MatrixAdd<Matrix<RM, Dim1>, T, T> for Matrix<SM, Dim1>
where
    T: Num,
    RM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
{
    fn add(&mut self, rhs: &Matrix<RM, Dim1>, lhs: &T) {
        assert_eq!(self.shape_stride().shape(), rhs.shape_stride().shape());

        for idx in 0..self.shape_stride().shape().num_elm() {
            self.index_item_asign(dim!(idx), rhs.index_item(dim!(idx)) + *lhs);
        }
    }
}

macro_rules! impl_matrix_add_scalar {
    ($dim:ty) => {
        impl<T, RM, SM> MatrixAdd<Matrix<RM, $dim>, T, T> for Matrix<SM, $dim>
        where
            T: Num,
            RM: ViewMemory<Item = T>,
            SM: ViewMutMemory<Item = T>,
        {
            fn add(&mut self, rhs: &Matrix<RM, $dim>, lhs: &T) {
                assert_eq!(self.shape_stride().shape(), rhs.shape_stride().shape());

                for idx in 0..self.shape_stride().shape()[0] {
                    self.index_axis_mut(Index0D::new(idx))
                        .add(&rhs.index_axis(Index0D::new(idx)), lhs);
                }
            }
        }
    };
}
impl_matrix_add_scalar!(Dim2);
impl_matrix_add_scalar!(Dim3);
impl_matrix_add_scalar!(Dim4);

// matrix add matrix 1d
impl<T, RM, LM, SM> MatrixAdd<Matrix<RM, Dim1>, Matrix<LM, Dim1>, T> for Matrix<SM, Dim1>
where
    T: Num,
    RM: ViewMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
{
    fn add(&mut self, rhs: &Matrix<RM, Dim1>, lhs: &Matrix<LM, Dim1>) {
        assert_eq!(self.shape_stride().shape(), rhs.shape_stride().shape());
        assert_eq!(self.shape_stride().shape(), lhs.shape_stride().shape());

        self.copy_from(rhs);

        <LM as Memory>::Blas::axpy(
            self.shape_stride().shape().num_elm(),
            T::one(),
            lhs.as_ptr(),
            lhs.shape_stride().stride()[0],
            self.as_mut_ptr(),
            self.shape_stride().stride()[0],
        );
    }
}

// matrix add for 2d
impl<T, RM, LM, SM> MatrixAdd<Matrix<RM, Dim2>, Matrix<LM, Dim2>, T> for Matrix<SM, Dim2>
where
    T: Num,
    RM: ViewMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
{
    fn add(&mut self, rhs: &Matrix<RM, Dim2>, lhs: &Matrix<LM, Dim2>) {
        assert_eq!(self.shape_stride().shape(), rhs.shape_stride().shape());
        assert_eq!(self.shape_stride().shape(), lhs.shape_stride().shape());

        for idx in 0..self.shape_stride().shape()[0] {
            let mut self_ = self.index_axis_mut(Index0D::new(idx));
            let rhs_ = rhs.index_axis(Index0D::new(idx));
            let lhs_ = lhs.index_axis(Index0D::new(idx));
            self_.add(&rhs_, &lhs_);
        }
    }
}

// matrix add for 3d
impl<T, RM, LM, SM> MatrixAdd<Matrix<RM, Dim3>, Matrix<LM, Dim3>, T> for Matrix<SM, Dim3>
where
    T: Num,
    RM: ViewMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
{
    fn add(&mut self, rhs: &Matrix<RM, Dim3>, lhs: &Matrix<LM, Dim3>) {
        assert_eq!(self.shape_stride().shape(), rhs.shape_stride().shape());
        assert_eq!(self.shape_stride().shape(), lhs.shape_stride().shape());

        for idx in 0..self.shape_stride().shape()[0] {
            let mut self_ = self.index_axis_mut(Index0D::new(idx));
            let rhs_ = rhs.index_axis(Index0D::new(idx));
            let lhs_ = lhs.index_axis(Index0D::new(idx));
            self_.add(&rhs_, &lhs_);
        }
    }
}

// matrix add for 4d
impl<T, RM, LM, SM> MatrixAdd<Matrix<RM, Dim4>, Matrix<LM, Dim4>, T> for Matrix<SM, Dim4>
where
    T: Num,
    RM: ViewMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
{
    fn add(&mut self, rhs: &Matrix<RM, Dim4>, lhs: &Matrix<LM, Dim4>) {
        assert_eq!(self.shape_stride().shape(), rhs.shape_stride().shape());
        assert_eq!(self.shape_stride().shape(), lhs.shape_stride().shape());

        for idx in 0..self.shape_stride().shape()[0] {
            let mut self_ = self.index_axis_mut(Index0D::new(idx));
            let rhs_ = rhs.index_axis(Index0D::new(idx));
            let lhs_ = lhs.index_axis(Index0D::new(idx));
            self_.add(&rhs_, &lhs_);
        }
    }
}

macro_rules! impl_matrix_add_matrix {
    ($dim_self:ty, $dim_lhs:ty) => {
        impl<'a, T, LM, RM, SM> MatrixAdd<Matrix<RM, $dim_self>, Matrix<LM, $dim_lhs>, T>
            for Matrix<SM, $dim_self>
        where
            T: Num,
            LM: ViewMemory<Item = T> + 'a,
            RM: ViewMemory<Item = T> + 'a,
            SM: ViewMutMemory<Item = T> + 'a,
        {
            fn add(&mut self, rhs: &Matrix<RM, $dim_self>, lhs: &Matrix<LM, $dim_lhs>) {
                assert_eq!(self.shape_stride().shape(), rhs.shape_stride().shape());

                for idx in 0..self.shape_stride().shape()[0] {
                    let mut self_ = self.index_axis_mut(Index0D::new(idx));
                    let rhs_ = rhs.index_axis(Index0D::new(idx));
                    self_.add(&rhs_, lhs);
                }
            }
        }
    };
}
impl_matrix_add_matrix!(Dim2, Dim1);
impl_matrix_add_matrix!(Dim3, Dim1);
impl_matrix_add_matrix!(Dim4, Dim1);
impl_matrix_add_matrix!(Dim3, Dim2);
impl_matrix_add_matrix!(Dim4, Dim2);
impl_matrix_add_matrix!(Dim4, Dim3);

#[cfg(test)]
mod add {
    use crate::{
        dim,
        matrix::{MatrixSlice, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::{CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix3D},
        operation::zeros::Zeros,
        slice,
    };

    use super::*;

    #[test]
    fn add_1d_scalar_default_stride() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], dim!(3));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(3));
        ans.to_view_mut().add(&a.to_view(), &1.0);

        assert_eq!(ans.index_item(dim!(0)), 2.0);
        assert_eq!(ans.index_item(dim!(1)), 3.0);
        assert_eq!(ans.index_item(dim!(2)), 4.0);
    }

    #[test]
    fn add_1d_scalar_sliced() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dim!(6));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(3));

        let sliced = a.slice(slice!(..;2));

        ans.to_view_mut().add(&sliced.to_view(), &1.0);

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

        ans.to_view_mut().add(&sliced.to_view(), &1.0);

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
        ans.to_view_mut().add(&a.to_view(), &b.to_view());

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
            .add(&sliced_a.to_view(), &sliced_b.to_view());

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
            .add(&sliced_a.to_view(), &sliced_b.to_view());

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
            .add(&sliced_a.to_view(), &sliced_b.to_view());

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
        ans.to_view_mut().add(&a.to_view(), &b.to_view());

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

use crate::{
    blas::{Blas, BlasLayout},
    dim::DimTrait,
    dim_impl::{Dim1, Dim2, Dim3, Dim4},
    element_wise::ElementWise,
    index::get_blas_trans_for_gemm,
    index_impl::Index0D,
    matrix::{AsMutPtr, AsPtr, IndexAxis, IndexAxisMut, MatrixBase, ViewMatrix, ViewMutMatix},
    matrix_impl::Matrix,
    memory::{ViewMemory, ViewMutMemory},
    num::Num,
    operation::transpose::Transpose,
};

use super::copy_from::CopyFrom;

pub trait Mul<Rhs, Lhs>: ViewMutMatix {
    fn mul(&mut self, rhs: &Rhs, lhs: &Lhs);
}

pub trait MatMul<Rhs, Lhs>: ViewMutMatix {
    fn mat_mul(&mut self, rhs: &Rhs, lhs: &Lhs);
}

impl<T, VM, R> Mul<R, T> for VM
where
    T: Num,
    VM: ViewMutMatix<Dim = Dim1, Item = T> + CopyFrom<R, T>,
    R: ViewMatrix<Dim = Dim1>,
{
    fn mul(&mut self, rhs: &R, lhs: &T) {
        assert_eq!(self.shape(), rhs.shape());
        self.copy_from(rhs);
        Self::Blas::scal(
            self.shape().num_elm(),
            *lhs,
            self.as_mut_ptr(),
            self.stride()[0],
        );
    }
}

impl<T, VM, R, L> Mul<Matrix<R, Dim1>, Matrix<L, Dim1>> for Matrix<VM, Dim1>
where
    T: Num,
    VM: ViewMutMemory<Item = T>,
    R: ViewMemory<Item = T>,
    L: ViewMemory<Item = T>,
{
    fn mul(&mut self, rhs: &Matrix<R, Dim1>, lhs: &Matrix<L, Dim1>) {
        assert_eq!(self.shape(), rhs.shape());
        assert_eq!(self.shape(), lhs.shape());
        let self_stride = self.stride()[0];
        let self_size = self.shape().num_elm();

        VM::ElmentWise::mul(
            self.as_mut_ptr(),
            rhs.as_ptr(),
            lhs.as_ptr(),
            self_size,
            rhs.stride()[0],
            lhs.stride()[0],
            self_stride,
        );
    }
}

macro_rules! impl_mul_same_dim {
    ($dim:ty) => {
        impl<T, VM, R, L> Mul<Matrix<R, $dim>, Matrix<L, $dim>> for Matrix<VM, $dim>
        where
            T: Num,
            VM: ViewMutMemory<Item = T>,
            R: ViewMemory<Item = T>,
            L: ViewMemory<Item = T>,
        {
            fn mul(&mut self, rhs: &Matrix<R, $dim>, lhs: &Matrix<L, $dim>) {
                assert_eq!(self.shape(), rhs.shape());
                assert_eq!(self.shape(), lhs.shape());

                for i in 0..self.shape()[0] {
                    let mut self_ = self.index_axis_mut(Index0D::new(i));
                    let rhs_ = rhs.index_axis(Index0D::new(i));
                    let lhs_ = lhs.index_axis(Index0D::new(i));
                    self_.mul(&rhs_, &lhs_);
                }
            }
        }
    };
}
impl_mul_same_dim!(Dim2);
impl_mul_same_dim!(Dim3);
impl_mul_same_dim!(Dim4);

macro_rules! impl_mul_scalar {
    ($dim:ty) => {
        impl<T, VM, R> Mul<Matrix<R, $dim>, T> for Matrix<VM, $dim>
        where
            T: Num,
            VM: ViewMutMemory<Item = T>,
            R: ViewMemory<Item = T>,
        {
            fn mul(&mut self, rhs: &Matrix<R, $dim>, lhs: &T) {
                assert_eq!(self.shape(), rhs.shape());
                if self.shape_stride().is_contiguous() {
                    self.copy_from(rhs);
                    let num_dim = self.shape().len();
                    Self::Blas::scal(
                        self.shape().num_elm(),
                        *lhs,
                        self.as_mut_ptr(),
                        self.stride()[num_dim - 1],
                    );
                } else {
                    for i in 0..self.shape()[0] {
                        let mut self_ = self.index_axis_mut(Index0D::new(i));
                        let rhs_ = rhs.index_axis(Index0D::new(i));
                        self_.mul(&rhs_, lhs);
                    }
                }
            }
        }
    };
}
impl_mul_scalar!(Dim2);
impl_mul_scalar!(Dim3);
impl_mul_scalar!(Dim4);

macro_rules! impl_mul_matrix_matrix {
    ($dim_s:ty, $dim_l:ty) => {
        impl<T, S, R, L> Mul<Matrix<R, $dim_s>, Matrix<L, $dim_l>> for Matrix<S, $dim_s>
        where
            T: Num,
            S: ViewMutMemory<Item = T>,
            R: ViewMemory<Item = T>,
            L: ViewMemory<Item = T>,
        {
            fn mul(&mut self, rhs: &Matrix<R, $dim_s>, lhs: &Matrix<L, $dim_l>) {
                assert_eq!(self.shape(), rhs.shape());

                for i in 0..self.shape()[0] {
                    self.index_axis_mut(Index0D::new(i))
                        .mul(&rhs.index_axis(Index0D::new(i)), lhs);
                }
            }
        }
    };
}
impl_mul_matrix_matrix!(Dim2, Dim1);
impl_mul_matrix_matrix!(Dim3, Dim1);
impl_mul_matrix_matrix!(Dim4, Dim1);
impl_mul_matrix_matrix!(Dim3, Dim2);
impl_mul_matrix_matrix!(Dim4, Dim2);
impl_mul_matrix_matrix!(Dim4, Dim3);

impl<T, S, R, L> MatMul<Matrix<R, Dim2>, Matrix<L, Dim2>> for Matrix<S, Dim2>
where
    T: Num,
    S: ViewMutMemory<Item = T>,
    R: ViewMemory<Item = T>,
    L: ViewMemory<Item = T>,
{
    fn mat_mul(&mut self, rhs: &Matrix<R, Dim2>, lhs: &Matrix<L, Dim2>) {
        assert_eq!(self.shape()[0], rhs.shape()[0]);
        assert_eq!(self.shape()[1], lhs.shape()[1]);
        assert_eq!(rhs.shape()[1], lhs.shape()[0]);

        assert!(self.shape_stride().is_contiguous());
        assert!(rhs.shape_stride().is_contiguous());
        assert!(lhs.shape_stride().is_contiguous());

        let m = self.shape()[0];
        let n = self.shape()[1];
        let k = rhs.shape()[1];

        let self_stride = self.shape()[1];
        let rhs_stride = rhs.shape()[1];
        let lhs_stride = lhs.shape()[1];

        let self_ptr = self.as_mut_ptr();
        let rhs_ptr = rhs.as_ptr();
        let lhs_ptr = lhs.as_ptr();

        Self::Blas::gemm(
            BlasLayout::RowMajor,
            get_blas_trans_for_gemm(rhs.shape_stride()),
            get_blas_trans_for_gemm(lhs.shape_stride()),
            m,
            n,
            k,
            T::one(),
            rhs_ptr,
            rhs_stride,
            lhs_ptr,
            lhs_stride,
            T::zero(),
            self_ptr,
            self_stride,
        );

        self.transpose();
    }
}

#[cfg(test)]
mod mul {
    use crate::{
        dim,
        matrix::{IndexItem, MatrixSlice, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::{CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix4D},
        operation::zeros::Zeros,
        slice,
    };

    use super::Mul;

    #[test]
    fn scalar_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3.], dim!(3));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(3));
        ans.to_view_mut().mul(&a.to_view(), &2.);

        assert_eq!(ans.index_item(dim!(0)), 2.);
        assert_eq!(ans.index_item(dim!(1)), 4.);
        assert_eq!(ans.index_item(dim!(2)), 6.);
    }

    #[test]
    fn sliced_scalar_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], dim!(4));
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros(dim!(2));
        ans.to_view_mut().mul(&a.to_view().slice(slice!(..;2)), &2.);

        assert_eq!(ans.index_item(dim!(0)), 2.);
        assert_eq!(ans.index_item(dim!(1)), 6.);
    }

    #[test]
    fn scalar_2d() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], dim![2, 3]);
        let mut ans = CpuOwnedMatrix2D::<f32>::zeros(dim![2, 3]);
        ans.to_view_mut().mul(&a.to_view(), &2.);

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
        ans.to_view_mut().mul(&a.to_view(), &b.to_view());

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
            &a.to_view().slice(slice!(..;2)),
            &b.to_view().slice(slice!(..;2)),
        );

        assert_eq!(ans.index_item(dim!(0)), 1.);
        assert_eq!(ans.index_item(dim!(1)), 9.);
    }

    #[test]
    fn default_2d_2d() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], dim![2, 3]);
        let b = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], dim![2, 3]);
        let mut ans = CpuOwnedMatrix2D::<f32>::zeros(dim![2, 3]);
        ans.to_view_mut().mul(&a.to_view(), &b.to_view());

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

        ans.to_view_mut().mul(&a.to_view(), &b.to_view());

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
        let a = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], dim![2, 3]);
        let b = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], dim![3, 2]);
        let mut ans = CpuOwnedMatrix2D::<f32>::zeros(dim![2, 2]);

        ans.to_view_mut().mat_mul(&a.to_view(), &b.to_view());
        dbg!(ans.index_item(dim!(0, 0)));
        dbg!(ans.index_item(dim!(0, 1)));
        dbg!(ans.index_item(dim!(1, 0)));
        dbg!(ans.index_item(dim!(1, 1)));
        assert_eq!(ans.index_item(dim!(0, 0)), 22.);
        assert_eq!(ans.index_item(dim!(0, 1)), 28.);
        assert_eq!(ans.index_item(dim!(1, 0)), 49.);
        assert_eq!(ans.index_item(dim!(1, 1)), 64.);
    }
}

use crate::{
    blas::Blas,
    dim::DimTrait,
    dim_impl::{Dim1, Dim2, Dim3, Dim4},
    index_impl::Index0D,
    matrix::{AsMutPtr, AsPtr, IndexAxis, IndexAxisMut, MatrixBase, ViewMatrix, ViewMutMatix},
    matrix_impl::Matrix,
    memory::{Memory, ViewMemory, ViewMutMemory},
    num::Num,
};

pub trait CopyFrom<RHS, T: Num>: ViewMutMatix
where
    RHS: ViewMatrix,
{
    fn copy_from(&mut self, rhs: &RHS);
}

impl<V: ViewMemory + Memory<Item = T>, VM: ViewMutMemory + Memory<Item = T>, T: Num>
    CopyFrom<Matrix<V, Dim1>, T> for Matrix<VM, Dim1>
{
    fn copy_from(&mut self, rhs: &Matrix<V, Dim1>) {
        assert_eq!(self.shape_stride().shape(), rhs.shape_stride().shape());

        <V as Memory>::Blas::copy(
            self.shape_stride().shape()[0],
            rhs.as_ptr(),
            rhs.shape_stride().stride()[0],
            self.as_mut_ptr() as *mut _,
            self.shape_stride().stride()[0],
        );
    }
}

macro_rules! impl_copy_from {
    ($dim:ty) => {
        impl<V: ViewMemory + Memory<Item = T>, VM: ViewMutMemory + Memory<Item = T>, T: Num>
            CopyFrom<Matrix<V, $dim>, T> for Matrix<VM, $dim>
        {
            fn copy_from(&mut self, rhs: &Matrix<V, $dim>) {
                assert_eq!(self.shape_stride().shape(), rhs.shape_stride().shape());

                if self.shape_stride().is_contiguous() && rhs.shape_stride().is_contiguous() {
                    let num_dims = self.shape_stride().shape().len();
                    let num_elms = self.shape_stride().shape().num_elm();
                    let self_stride = self.shape_stride().stride()[num_dims - 1];
                    let rhs_stride = rhs.shape_stride().stride()[num_dims - 1];

                    <V as Memory>::Blas::copy(
                        num_elms,
                        rhs.as_ptr(),
                        rhs_stride,
                        self.as_mut_ptr() as *mut _,
                        self_stride,
                    );
                } else {
                    for i in 0..self.shape_stride().shape()[0] {
                        let rhs = rhs.index_axis(Index0D::new(i));
                        let mut self_ = self.index_axis_mut(Index0D::new(i));

                        self_.copy_from(&rhs);
                    }
                }
            }
        }
    };
}
impl_copy_from!(Dim2);
impl_copy_from!(Dim3);
impl_copy_from!(Dim4);

#[cfg(test)]
mod deep_copy {
    use super::*;
    use crate::{
        dim,
        matrix::{
            IndexItem, MatrixSlice, MatrixSliceMut, OwnedMatrix, ToViewMatrix, ToViewMutMatrix,
        },
        matrix_impl::{CpuOwnedMatrix1D, CpuOwnedMatrix2D},
        slice,
    };

    #[test]
    fn default_stride_1d() {
        let a = vec![0f32; 6];
        let b = vec![1f32, 2., 3., 4., 5., 6.];

        let mut a = CpuOwnedMatrix1D::from_vec(a, dim!(6));
        let b = CpuOwnedMatrix1D::from_vec(b, dim!(6));

        a.to_view_mut().copy_from(&b.to_view());

        assert_eq!(a.index_item(dim!(0)), 1.);
        assert_eq!(a.index_item(dim!(1)), 2.);
        assert_eq!(a.index_item(dim!(2)), 3.);
        assert_eq!(a.index_item(dim!(3)), 4.);
        assert_eq!(a.index_item(dim!(4)), 5.);
        assert_eq!(a.index_item(dim!(5)), 6.);
    }

    #[test]
    fn sliced_1d() {
        let a = vec![0f32; 6];
        let v = vec![0f32, 1., 2., 3., 4., 5.];

        let mut a = CpuOwnedMatrix1D::from_vec(a.clone(), dim!(6));
        let v = CpuOwnedMatrix1D::from_vec(v, dim!(6));

        let mut a_sliced = a.slice_mut(slice!(..;2));
        let v_sliced = v.slice(slice!(0..3));

        a_sliced.copy_from(&v_sliced);
        assert_eq!(a.index_item(dim!(0)), 0.);
        assert_eq!(a.index_item(dim!(1)), 0.);
        assert_eq!(a.index_item(dim!(2)), 1.);
        assert_eq!(a.index_item(dim!(3)), 0.);
        assert_eq!(a.index_item(dim!(4)), 2.);
        assert_eq!(a.index_item(dim!(5)), 0.);
    }

    #[test]
    fn defualt_stride_2d() {
        let a = vec![0f32; 6];
        let b = vec![1f32, 2., 3., 4., 5., 6.];

        let mut a = CpuOwnedMatrix2D::from_vec(a, dim!(2, 3));
        let b = CpuOwnedMatrix2D::from_vec(b, dim!(2, 3));

        a.to_view_mut().copy_from(&b.to_view());

        assert_eq!(a.index_item(dim!(0, 0)), 1.);
        assert_eq!(a.index_item(dim!(0, 1)), 2.);
        assert_eq!(a.index_item(dim!(0, 2)), 3.);
        assert_eq!(a.index_item(dim!(1, 0)), 4.);
        assert_eq!(a.index_item(dim!(1, 1)), 5.);
        assert_eq!(a.index_item(dim!(1, 2)), 6.);
    }

    #[test]
    fn sliced_2d() {
        let a = vec![0f32; 12];
        let v = vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.];
        let mut a = CpuOwnedMatrix2D::from_vec(a.clone(), dim!(3, 4));
        let v = CpuOwnedMatrix2D::from_vec(v, dim!(3, 4));

        let mut a_sliced = a.slice_mut(slice!(0..2, 0..3));
        let v_sliced = v.slice(slice!(1..3, 1..4));

        a_sliced.copy_from(&v_sliced);
        assert_eq!(a.index_item(dim!(0, 0)), 5.);
        assert_eq!(a.index_item(dim!(0, 1)), 6.);
        assert_eq!(a.index_item(dim!(0, 2)), 7.);
        assert_eq!(a.index_item(dim!(0, 3)), 0.);
        assert_eq!(a.index_item(dim!(1, 0)), 9.);
        assert_eq!(a.index_item(dim!(1, 1)), 10.);
        assert_eq!(a.index_item(dim!(1, 2)), 11.);
        assert_eq!(a.index_item(dim!(2, 3)), 0.);
    }
}

use ruml_index_impl::index::{Index0D, Index1D, Index2D, Index3D};
use ruml_matrix_traits::{
    matrix::{IndexAxis, ViewMatrix},
    memory::{Memory, OwnedMemory, ViewMemory},
    num::Num,
};

use crate::matrix::{
    CpuOwnedMatrix2D, CpuOwnedMatrix3D, CpuOwnedMatrix4D, CpuViewMatrix1D, CpuViewMatrix2D,
    CpuViewMatrix3D, CpuViewMatrix4D,
};

macro_rules! impl_index_axis {
    ($impl_ty:ty, $output_ty:ty, $($index_ty:ty)*) => {
        $(
            impl<T: Num> IndexAxis<$index_ty> for $impl_ty {
                type Output<'a> = $output_ty where Self: 'a;

                fn index_axis(&self, index: $index_ty) -> Self::Output<'_> {
                    let shape_stride = index.get_shape_stride(&self.shape(), &self.stride());
                    let offset = index.get_offset(self.stride());
                    let data = self.data().to_view(offset);
                    Self::Output::construct(data, shape_stride.shape(), shape_stride.stride())
                }
            }
        )*
    };
}

impl_index_axis!(CpuOwnedMatrix2D<T>, CpuViewMatrix1D<'a, T>, Index0D Index1D);
impl_index_axis!(CpuOwnedMatrix3D<T>, CpuViewMatrix2D<'a, T>, Index0D Index1D Index2D);
impl_index_axis!(CpuOwnedMatrix4D<T>, CpuViewMatrix3D<'a, T>, Index0D Index1D Index2D Index3D);

macro_rules! impl_index_view_axis {
    ($impl_ty:ty, $output_ty:ty, $($index_ty:ty)*) => {
        $(
            impl<'a, T: Num> IndexAxis<$index_ty> for $impl_ty {
                type Output<'b> = $output_ty where Self: 'b;

                fn index_axis(&self, index: $index_ty) -> Self::Output<'_> {
                    let shape_stride = index.get_shape_stride(&self.shape(), &self.stride());
                    let offset = index.get_offset(self.stride()) + self.data().get_offset();
                    let mut data = self.data().clone();
                    data.set_offset(offset);
                    Self::Output::construct(data, shape_stride.shape(), shape_stride.stride())
                }
            }
        )*
    };
}

impl_index_view_axis!(CpuViewMatrix2D<'a, T>, CpuViewMatrix1D<'a, T>, Index0D Index1D);
impl_index_view_axis!(CpuViewMatrix3D<'a, T>, CpuViewMatrix2D<'a, T>, Index0D Index1D Index2D);
impl_index_view_axis!(CpuViewMatrix4D<'a, T>, CpuViewMatrix3D<'a, T>, Index0D Index1D Index2D Index3D);

#[cfg(test)]
mod index_axis {
    use ruml_dim_impl::{Dim1, Dim2, Dim3};
    use ruml_index_impl::index::{Index0D, Index1D};
    use ruml_matrix_traits::matrix::{IndexAxis, OwnedMatrix};

    use crate::matrix::{CpuOwnedMatrix2D, CpuOwnedMatrix3D};

    #[test]
    fn matrix_2d_0d_index() {
        let owned = CpuOwnedMatrix2D::from_vec(vec![0., 1., 2., 3.], Dim2::new([2, 2]));
        let index_0d_axis = owned.index_axis(Index0D(1));
        assert_eq!(index_0d_axis.shape(), Dim1::new([2]));
        assert_eq!(index_0d_axis.stride(), Dim1::new([1]));

        assert_eq!(index_0d_axis[Dim1::new([0])], 2.);
        assert_eq!(index_0d_axis[Dim1::new([1])], 3.);
    }

    #[test]
    fn matrix_2d_1d_index() {
        let owned = CpuOwnedMatrix2D::from_vec(vec![0., 1., 2., 3.], Dim2::new([2, 2]));
        let index_1d_axis = owned.index_axis(Index1D(1));
        assert_eq!(index_1d_axis.shape(), Dim1::new([2]));
        assert_eq!(index_1d_axis.stride(), Dim1::new([2]));

        assert_eq!(index_1d_axis[Dim1::new([0])], 1.);
        assert_eq!(index_1d_axis[Dim1::new([1])], 3.);
    }

    #[test]
    fn matrix_3d_axis_0() {
        let mut v = Vec::new();
        for i in 0..3 * 3 * 3 {
            v.push(i as f32);
        }
        let owned = CpuOwnedMatrix3D::from_vec(v, Dim3::new([3, 3, 3]));
        let index_0d_axis = owned.index_axis(Index0D(2));

        assert_eq!(index_0d_axis.shape(), Dim2::new([3, 3]));
        assert_eq!(index_0d_axis.stride(), Dim2::new([3, 1]));

        assert_eq!(index_0d_axis[Dim2::new([0, 0])], 18.);
        assert_eq!(index_0d_axis[Dim2::new([0, 1])], 19.);
        assert_eq!(index_0d_axis[Dim2::new([0, 2])], 20.);
        assert_eq!(index_0d_axis[Dim2::new([1, 0])], 21.);
        assert_eq!(index_0d_axis[Dim2::new([1, 1])], 22.);
        assert_eq!(index_0d_axis[Dim2::new([1, 2])], 23.);
        assert_eq!(index_0d_axis[Dim2::new([2, 0])], 24.);
        assert_eq!(index_0d_axis[Dim2::new([2, 1])], 25.);
        assert_eq!(index_0d_axis[Dim2::new([2, 2])], 26.);
    }

    #[test]
    fn matrix_3d_axis_1() {
        let mut v = Vec::new();
        for i in 0..3 * 3 * 3 {
            v.push(i as f32);
        }
        let owned = CpuOwnedMatrix3D::from_vec(v, Dim3::new([3, 3, 3]));
        let index_1d_axis = owned.index_axis(Index1D(2));

        assert_eq!(index_1d_axis.shape(), Dim2::new([3, 3]));
        assert_eq!(index_1d_axis.stride(), Dim2::new([9, 1]));

        assert_eq!(index_1d_axis[Dim2::new([0, 0])], 6.);
        assert_eq!(index_1d_axis[Dim2::new([0, 1])], 7.);
        assert_eq!(index_1d_axis[Dim2::new([0, 2])], 8.);
        assert_eq!(index_1d_axis[Dim2::new([1, 0])], 15.);
        assert_eq!(index_1d_axis[Dim2::new([1, 1])], 16.);
        assert_eq!(index_1d_axis[Dim2::new([1, 2])], 17.);
        assert_eq!(index_1d_axis[Dim2::new([2, 0])], 24.);
        assert_eq!(index_1d_axis[Dim2::new([2, 1])], 25.);
        assert_eq!(index_1d_axis[Dim2::new([2, 2])], 26.);
    }

    #[test]
    fn matrix_3d_axis_2() {
        let mut v = Vec::new();
        for i in 0..3 * 3 * 3 {
            v.push(i as f32);
        }
        let owned = CpuOwnedMatrix3D::from_vec(v, Dim3::new([3, 3, 3]));
        let index_2d_axis = owned.index_axis(Index1D(2));

        assert_eq!(index_2d_axis.shape(), Dim2::new([3, 3]));
        assert_eq!(index_2d_axis.stride(), Dim2::new([9, 1]));

        assert_eq!(index_2d_axis[Dim2::new([0, 0])], 6.);
        assert_eq!(index_2d_axis[Dim2::new([0, 1])], 7.);
        assert_eq!(index_2d_axis[Dim2::new([0, 2])], 8.);
        assert_eq!(index_2d_axis[Dim2::new([1, 0])], 15.);
        assert_eq!(index_2d_axis[Dim2::new([1, 1])], 16.);
        assert_eq!(index_2d_axis[Dim2::new([1, 2])], 17.);
        assert_eq!(index_2d_axis[Dim2::new([2, 0])], 24.);
        assert_eq!(index_2d_axis[Dim2::new([2, 1])], 25.);
        assert_eq!(index_2d_axis[Dim2::new([2, 2])], 26.);
    }
}

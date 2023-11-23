use ruml_index_impl::slice::{Slice1D, Slice2D, Slice3D, Slice4D};
use ruml_matrix_traits::{index::SliceTrait, matrix::MatrixSlice, memory::OwnedMemory, num::Num};

use crate::matrix::{
    CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix3D, CpuOwnedMatrix4D, CpuViewMatrix1D,
    CpuViewMatrix2D, CpuViewMatrix3D, CpuViewMatrix4D,
};

macro_rules! impl_slice {
    ($owned_ty:ty, $view_ty:ty, $slice_ty:ty) => {
        impl<T: Num> MatrixSlice<$slice_ty> for $owned_ty {
            type Output = $view_ty;

            fn slice(&self, index: $slice_ty) -> Self::Output {
                let shape = self.shape();
                let stride = self.stride();

                let new_shape_stride = index.sliced_shape_stride(&shape, &stride);

                let data = self.data().to_view(0);
                // let data = Arc::new(data);

                <$view_ty>::new(data, new_shape_stride.shape(), new_shape_stride.stride())
            }
        }

        impl<T: Num> MatrixSlice<$slice_ty> for $view_ty {
            type Output = $view_ty;

            fn slice(&self, index: $slice_ty) -> Self::Output {
                let shape = self.shape();
                let stride = self.stride();

                let new_shape_stride = index.sliced_shape_stride(&shape, &stride);

                let data = self.data().clone();

                <$view_ty>::new(data, new_shape_stride.shape(), new_shape_stride.stride())
            }
        }
    };
}

impl_slice!(CpuOwnedMatrix1D<T>, CpuViewMatrix1D<T>, Slice1D);
impl_slice!(CpuOwnedMatrix2D<T>, CpuViewMatrix2D<T>, Slice2D);
impl_slice!(CpuOwnedMatrix3D<T>, CpuViewMatrix3D<T>, Slice3D);
impl_slice!(CpuOwnedMatrix4D<T>, CpuViewMatrix4D<T>, Slice4D);

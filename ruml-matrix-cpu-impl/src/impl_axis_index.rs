use ruml_index_impl::index::{Index0D, Index1D, Index2D, Index3D};
use ruml_matrix_traits::{
    matrix::{IndexAxis, ViewMatrix},
    memory::OwnedMemory,
    num::Num,
};

use crate::matrix::{
    CpuOwnedMatrix2D, CpuOwnedMatrix3D, CpuOwnedMatrix4D, CpuViewMatrix1D, CpuViewMatrix2D,
    CpuViewMatrix3D, CpuViewMatrix4D,
};
use crate::memory::CpuViewMemory;

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
                    let offset = index.get_offset(self.stride());
                    let data = CpuViewMemory::new(self.data().reference(), offset);
                    Self::Output::construct(data, shape_stride.shape(), shape_stride.stride())
                }
            }
        )*
    };
}

impl_index_view_axis!(CpuViewMatrix2D<'a, T>, CpuViewMatrix1D<'a, T>, Index0D Index1D);
impl_index_view_axis!(CpuViewMatrix3D<'a, T>, CpuViewMatrix2D<'a, T>, Index0D Index1D Index2D);
impl_index_view_axis!(CpuViewMatrix4D<'a, T>, CpuViewMatrix3D<'a, T>, Index0D Index1D Index2D Index3D);

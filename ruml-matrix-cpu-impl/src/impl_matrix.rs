use ruml_dim_impl::{Dim1, Dim2, Dim3, Dim4};
use ruml_matrix_traits::{
    index::ShapeStride,
    matrix::{Matrix, OwnedMatrix, ViewMatrix},
    memory::{OwnedMemory, ViewMemory},
    num::Num,
};

use crate::{
    matrix::{
        CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix3D, CpuOwnedMatrix4D, CpuViewMatrix1D,
        CpuViewMatrix2D, CpuViewMatrix3D, CpuViewMatrix4D,
    },
    memory::{CpuOwnedMemory, CpuViewMemory},
};

macro_rules! impl_matrix {
    ($impl_ty:ty, $dim:ty, $memory:ty) => {
        impl<T: Num> Matrix for $impl_ty {
            type Dim = $dim;
            type Memory = $memory;

            fn shape_stride(&self) -> ShapeStride<Self::Dim> {
                ShapeStride::new(self.shape(), self.stride())
            }

            fn memory(&self) -> &Self::Memory {
                &self.data()
            }
        }
    };
}

impl_matrix!(CpuOwnedMatrix1D<T>, Dim1, CpuOwnedMemory<T>);
impl_matrix!(CpuViewMatrix1D<T>, Dim1, CpuViewMemory<T>);
impl_matrix!(CpuOwnedMatrix2D<T>, Dim2, CpuOwnedMemory<T>);
impl_matrix!(CpuViewMatrix2D<T>, Dim2, CpuViewMemory<T>);
impl_matrix!(CpuOwnedMatrix3D<T>, Dim3, CpuOwnedMemory<T>);
impl_matrix!(CpuViewMatrix3D<T>, Dim3, CpuViewMemory<T>);
impl_matrix!(CpuOwnedMatrix4D<T>, Dim4, CpuOwnedMemory<T>);
impl_matrix!(CpuViewMatrix4D<T>, Dim4, CpuViewMemory<T>);

macro_rules! impl_owned {
    ($owned_ty:ty, $view_ty:ty) => {
        impl<T: Num> OwnedMatrix for $owned_ty {
            type View = $view_ty;

            fn to_view(&self) -> Self::View {
                let data = self.data().to_view(0);

                Self::View::new(data, self.shape(), self.stride())
            }

            fn construct(data: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self {
                Self::new(data, shape, stride)
            }
        }

        impl<T: Num> ViewMatrix for $view_ty {
            type Owned = $owned_ty;

            fn construct(data: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self {
                Self::new(data, shape, stride)
            }

            fn to_owned(&self) -> Self::Owned {
                let data: &CpuViewMemory<T> = self.data();
                let owned_data = <Self::Memory as ViewMemory>::to_owned(data);

                Self::Owned::construct(owned_data, self.shape(), self.stride())
            }
        }
    };
}

impl_owned!(CpuOwnedMatrix1D<T>, CpuViewMatrix1D<T>);
impl_owned!(CpuOwnedMatrix2D<T>, CpuViewMatrix2D<T>);
impl_owned!(CpuOwnedMatrix3D<T>, CpuViewMatrix3D<T>);
impl_owned!(CpuOwnedMatrix4D<T>, CpuViewMatrix4D<T>);

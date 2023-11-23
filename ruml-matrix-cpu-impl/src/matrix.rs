use ruml_dim_impl::{Dim1, Dim2, Dim3, Dim4};
use ruml_matrix_traits::num::Num;

use crate::memory::{CpuOwnedMemory, CpuViewMemory};

#[derive(Clone)]
pub struct CpuOwnedMatrix1D<T: Num> {
    data: CpuOwnedMemory<T>,
    shape: Dim1,
    stride: Dim1,
}

#[derive(Clone)]
pub struct CpuOwnedMatrix2D<T: Num> {
    data: CpuOwnedMemory<T>,
    shape: Dim2,
    stride: Dim2,
}

#[derive(Clone)]
pub struct CpuOwnedMatrix3D<T: Num> {
    data: CpuOwnedMemory<T>,
    shape: Dim3,
    stride: Dim3,
}

#[derive(Clone)]
pub struct CpuOwnedMatrix4D<T: Num> {
    data: CpuOwnedMemory<T>,
    shape: Dim4,
    stride: Dim4,
}

#[derive(Clone)]
pub struct CpuViewMatrix1D<T: Num> {
    data: CpuViewMemory<T>,
    shape: Dim1,
    stride: Dim1,
}

#[derive(Clone)]
pub struct CpuViewMatrix2D<T: Num> {
    data: CpuViewMemory<T>,
    shape: Dim2,
    stride: Dim2,
}

#[derive(Clone)]
pub struct CpuViewMatrix3D<T: Num> {
    data: CpuViewMemory<T>,
    shape: Dim3,
    stride: Dim3,
}

#[derive(Clone)]
pub struct CpuViewMatrix4D<T: Num> {
    data: CpuViewMemory<T>,
    shape: Dim4,
    stride: Dim4,
}

macro_rules! impl_methods {
    ($tt:ty, $dim:ty, $data:ty) => {
        impl<T: Num> $tt {
            pub fn new(data: $data, shape: $dim, stride: $dim) -> Self {
                Self {
                    data,
                    shape,
                    stride,
                }
            }

            pub fn shape(&self) -> $dim {
                self.shape
            }

            pub fn stride(&self) -> $dim {
                self.stride
            }

            pub fn data(&self) -> &$data {
                &self.data
            }
        }
    };
}

impl_methods!(CpuOwnedMatrix1D<T>, Dim1, CpuOwnedMemory<T>);
impl_methods!(CpuOwnedMatrix2D<T>, Dim2, CpuOwnedMemory<T>);
impl_methods!(CpuOwnedMatrix3D<T>, Dim3, CpuOwnedMemory<T>);
impl_methods!(CpuOwnedMatrix4D<T>, Dim4, CpuOwnedMemory<T>);
impl_methods!(CpuViewMatrix1D<T>, Dim1, CpuViewMemory<T>);
impl_methods!(CpuViewMatrix2D<T>, Dim2, CpuViewMemory<T>);
impl_methods!(CpuViewMatrix3D<T>, Dim3, CpuViewMemory<T>);
impl_methods!(CpuViewMatrix4D<T>, Dim4, CpuViewMemory<T>);

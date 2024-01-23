use crate::{
    dim_impl::{Dim2, Dim3, Dim4},
    matrix::MatrixBase,
    matrix_impl::Matrix,
    memory::Memory,
    num::Num,
};

pub trait Transpose {
    fn transpose(&mut self);
}

macro_rules! impl_transpose {
    ($dim:ty) => {
        impl<T: Num, M: Memory<Item = T>> Transpose for Matrix<M, $dim> {
            #[allow(clippy::almost_swapped)]
            fn transpose(&mut self) {
                let shape_stride = self.shape_stride();
                let transposed = shape_stride.transpose();

                self.update_shape(transposed.shape());
                self.update_stride(transposed.stride());
            }
        }
    };
}
impl_transpose!(Dim2);
impl_transpose!(Dim3);
impl_transpose!(Dim4);

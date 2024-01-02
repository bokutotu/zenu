use crate::{
    dim::DimTrait,
    dim_impl::{Dim2, Dim3, Dim4},
    matrix::{MatrixBase, ViewMutMatix},
    matrix_impl::Matrix,
    memory::{Memory, ViewMutMemory},
    num::Num,
};

pub trait Transpose: ViewMutMatix {
    fn transpose(&mut self);
}

macro_rules! impl_transpose {
    ($dim:ty) => {
        impl<T: Num, M: ViewMutMemory + Memory<Item = T>> Transpose for Matrix<M, $dim> {
            fn transpose(&mut self) {
                let shape_stride = self.shape_stride();
                let mut shape = shape_stride.shape();
                let mut stride = shape_stride.stride();

                let num_dim = shape.len();

                shape[num_dim - 2] = shape[num_dim - 1];
                shape[num_dim - 1] = shape[num_dim - 2];

                stride[num_dim - 2] = stride[num_dim - 1];
                stride[num_dim - 1] = stride[num_dim - 2];

                self.update_shape(shape);
                self.update_stride(stride);
            }
        }
    };
}
impl_transpose!(Dim2);
impl_transpose!(Dim3);
impl_transpose!(Dim4);

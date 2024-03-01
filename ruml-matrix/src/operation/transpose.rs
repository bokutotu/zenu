use crate::{
    dim::{Dim2, Dim3, Dim4, DimDyn},
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

impl<T: Num, M: Memory<Item = T>> Transpose for Matrix<M, DimDyn> {
    fn transpose(&mut self) {
        let shape_stride = self.shape_stride();
        let transposed = shape_stride.transpose();

        self.update_shape(transposed.shape());
        self.update_stride(transposed.stride());
    }
}

#[cfg(test)]
mod transpose {
    use crate::{
        matrix::{IndexItem, OwnedMatrix},
        matrix_impl::OwnedMatrixDyn,
    };

    use super::Transpose;

    #[test]
    fn transpose_2d() {
        let mut a = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        a.transpose();
        assert_eq!(a.index_item([0, 0]), 1.);
        assert_eq!(a.index_item([0, 1]), 4.);
        assert_eq!(a.index_item([1, 0]), 2.);
        assert_eq!(a.index_item([1, 1]), 5.);
        assert_eq!(a.index_item([2, 0]), 3.);
        assert_eq!(a.index_item([2, 1]), 6.);
    }
}

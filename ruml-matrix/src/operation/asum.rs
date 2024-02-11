use crate::{
    blas::Blas,
    dim::{Dim1, Dim2, Dim3, Dim4, DimTrait},
    index::Index0D,
    matrix::{AsPtr, IndexAxis, MatrixBase},
    matrix_impl::Matrix,
    memory::{Memory, ToViewMemory},
    num::Num,
};

pub trait Asum<T> {
    fn asum(&self) -> T;
}

impl<M: Memory<Item = T>, T: Num> Asum<T> for Matrix<M, Dim1> {
    fn asum(&self) -> T {
        let num_elm = self.shape_stride().shape().num_elm();
        let stride = self.shape_stride().stride()[0];
        M::Blas::asum(num_elm, self.as_ptr(), stride)
    }
}

macro_rules! impl_asum {
    ($($dim:ty),*) => {
        $(
            impl<M: Memory<Item = T> + ToViewMemory, T: Num> Asum<T> for Matrix<M, $dim> {
                fn asum(&self) -> T {
        if self.shape_stride().is_contiguous() {
            let num_elm = self.shape_stride().shape().num_elm();
            let num_dim = self.shape_stride().shape().len();
            let stride = self.shape_stride().stride();
            return M::Blas::asum(num_elm, self.as_ptr(), stride[num_dim - 1]);
        } else {
            let mut sum = T::zero();
            for i in 0..self.shape_stride().shape()[0] {
                let tmp = self.index_axis(Index0D::new(i));
                sum = sum + tmp.asum();
            }
            return sum;
        }
                }
            }
        )*
    };
}
impl_asum!(Dim2);
impl_asum!(Dim3);
impl_asum!(Dim4);

#[cfg(test)]
mod asum {
    use crate::{
        dim,
        matrix::{MatrixSlice, OwnedMatrix},
        matrix_impl::{CpuOwnedMatrix1D, CpuOwnedMatrix2D},
        slice,
    };

    use super::*;

    #[test]
    fn defualt_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], dim!(3));
        assert_eq!(a.asum(), 6.0);
    }

    #[test]
    fn defualt_2d() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1.0, 2.0, 3.0, 4.0], dim!(2, 2));
        assert_eq!(a.asum(), 10.0);
    }

    #[test]
    fn sliced_2d() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1.0, 2.0, 3.0, 4.0], dim!(2, 2));
        let b = a.slice(slice!(0..2, 0..1));
        assert_eq!(b.asum(), 4.0);
    }
}

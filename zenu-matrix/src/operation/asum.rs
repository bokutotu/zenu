use crate::{
    blas::Blas,
    dim::DimTrait,
    index::Index0D,
    matrix::{AsPtr, IndexAxisDyn, MatrixBase},
    matrix_impl::Matrix,
    memory::ToViewMemory,
    num::Num,
};

pub trait Asum<T> {
    fn asum(self) -> T;
}

impl<T, M, D> Asum<T> for Matrix<M, D>
where
    T: Num,
    M: ToViewMemory<Item = T>,
    D: DimTrait,
{
    fn asum(self) -> M::Item {
        let s = self.into_dyn_dim();
        if s.shape().is_empty() {
            unsafe { *s.as_ptr() }
        } else if s.shape_stride().is_contiguous() {
            let num_elm = s.shape().num_elm();
            let num_dim = s.shape().len();
            let stride = s.stride();
            M::Blas::asum(num_elm, s.as_ptr(), stride[num_dim - 1])
        } else {
            let mut sum = T::zero();
            for i in 0..s.shape()[0] {
                let tmp = s.index_axis_dyn(Index0D::new(i));
                sum += tmp.asum();
            }
            sum
        }
    }
}

#[cfg(test)]
mod asum {
    use crate::{
        matrix::{MatrixSlice, OwnedMatrix},
        matrix_impl::{OwnedMatrix1D, OwnedMatrix2D},
        slice,
    };

    use super::*;

    #[test]
    fn defualt_1d() {
        let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        assert_eq!(a.asum(), 6.0);
    }

    #[test]
    fn defualt_2d() {
        let a = OwnedMatrix2D::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert_eq!(a.asum(), 10.0);
    }

    #[test]
    fn sliced_2d() {
        let a = OwnedMatrix2D::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b = a.slice(slice!(0..2, 0..1));
        assert_eq!(b.asum(), 4.0);
    }
}

use crate::{
    blas::Blas,
    dim::DimTrait,
    dim_impl::Dim1,
    matrix::{BlasMatrix, ViewMatrix},
    num::Num,
};

pub trait Dot<Other, T> {
    fn dot(&self, other: Other) -> T;
}

impl<T, S, O> Dot<O, T> for S
where
    T: Num,
    S: ViewMatrix<Item = T, Dim = Dim1>,
    O: ViewMatrix<Item = T, Dim = Dim1>,
{
    fn dot(&self, other: O) -> T {
        if self.shape_stride().shape() != other.shape_stride().shape() {
            panic!("shape and stride must be same");
        }
        let num_elm = self.shape_stride().shape().num_elm();
        let self_stride = self.shape_stride().stride()[0];
        let other_stride = other.shape_stride().stride()[0];
        <Self as BlasMatrix>::Blas::dot(
            num_elm,
            self.as_ptr(),
            self_stride,
            other.as_ptr(),
            other_stride,
        )
    }
}

#[cfg(test)]
mod dot {
    use crate::{
        dim,
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::CpuOwnedMatrix1D,
    };

    use super::Dot;

    #[test]
    fn dot() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], dim!(3));
        let b = CpuOwnedMatrix1D::from_vec(vec![4.0, 5.0, 6.0], dim!(3));
        let c = a.to_view().dot(b.to_view());

        assert_eq!(c, 32.0);
    }
}

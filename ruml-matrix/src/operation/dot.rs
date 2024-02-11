use crate::{
    dim::Dim1,
    matrix::{MatrixBase, ViewMatrix},
    matrix_blas::dot::dot,
    num::Num,
};

pub trait Dot<Other, T> {
    fn dot(self, other: Other) -> T;
}

impl<T, S, O> Dot<O, T> for S
where
    T: Num,
    S: ViewMatrix + MatrixBase<Dim = Dim1, Item = T>,
    O: ViewMatrix + MatrixBase<Dim = Dim1, Item = T>,
{
    fn dot(self, other: O) -> T {
        dot(self, other)
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

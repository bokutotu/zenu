use crate::{
    blas::Blas,
    dim::{Dim1, DimTrait},
    matrix::{MatrixBase, ViewMatrix},
    num::Num,
};

pub fn dot<T, X, Y>(x: X, y: Y) -> T
where
    T: Num,
    X: ViewMatrix + MatrixBase<Dim = Dim1, Item = T>,
    Y: ViewMatrix + MatrixBase<Dim = Dim1, Item = T>,
{
    assert_eq!(x.shape(), y.shape());
    assert_eq!(x.shape().len(), 1);
    dot_unchecked(x, y)
}

fn dot_unchecked<T, X, Y>(x: X, y: Y) -> T
where
    T: Num,
    X: ViewMatrix + MatrixBase<Dim = Dim1, Item = T>,
    Y: ViewMatrix + MatrixBase<Dim = Dim1, Item = T>,
{
    X::Blas::dot(
        x.shape()[0],
        x.as_ptr(),
        x.stride()[0],
        y.as_ptr(),
        y.stride()[0],
    )
}

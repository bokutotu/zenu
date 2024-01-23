use crate::{
    blas::Blas,
    dim_impl::Dim1,
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
    let mut x = x;

    X::Blas::dot(
        x.shape()[0],
        x.as_ptr(),
        x.stride()[0],
        y.as_ptr(),
        y.stride()[0],
    )
}

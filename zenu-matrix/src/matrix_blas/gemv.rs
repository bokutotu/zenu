use crate::{
    blas::{Blas, BlasLayout, BlasTrans},
    dim::{Dim1, Dim2},
    matrix::{AsMutPtr, AsPtr, MatrixBase, ViewMatrix, ViewMutMatix},
    num::Num,
};

pub fn gemv<T, A, Y, Z>(a: A, y: Y, z: Z, alpha: T, beta: T)
where
    T: Num,
    A: ViewMatrix + MatrixBase<Item = T, Dim = Dim2>,
    Y: ViewMatrix + MatrixBase<Item = T, Dim = Dim1>,
    Z: ViewMutMatix + MatrixBase<Item = T, Dim = Dim1>,
{
    if !a.shape_stride().is_contiguous() {
        panic!("a must be contiguous");
    }
    let mut z = z;
    let m = a.shape()[0];
    let n = a.shape()[1];
    let trans = if z.shape_stride().is_transposed() {
        BlasTrans::Ordinary
    } else {
        BlasTrans::None
    };
    let lda = if a.shape_stride().is_transposed() {
        m
    } else {
        n
    };
    let incx = a.stride()[0];
    let incy = y.stride()[0];
    assert_eq!(y.shape()[0], n);
    assert_eq!(z.shape()[0], m);
    let x = a.to_view();
    let y = y.to_view();
    let mut z = z.to_view_mut();
    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let z_ptr = z.as_mut_ptr();
    A::Blas::gemv(
        BlasLayout::ColMajor,
        trans,
        m,
        n,
        alpha,
        x_ptr,
        lda,
        y_ptr,
        incx,
        beta,
        z_ptr,
        incy,
    );
}

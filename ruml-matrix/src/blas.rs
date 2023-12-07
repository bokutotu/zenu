use crate::num::Num;

pub enum BlasTrans {
    None,
    Ordinary,
    Conjugate,
}

pub enum BlasLayout {
    RowMajor,
    ColMajor,
}

pub trait Blas<T: Num> {
    fn swap(n: usize, x: *mut T, incx: usize, y: *mut T, incy: usize);
    fn scal(n: usize, alpha: T, x: *mut T, incx: usize);
    fn copy(n: usize, x: *const T, incx: usize, y: *mut T, incy: usize);
    fn dot(n: usize, x: *mut T, incx: usize, y: *mut T, incy: usize) -> T;
    fn norm2(n: usize, x: *mut T, incx: usize) -> T;
    fn asum(n: usize, x: *mut T, incx: usize) -> T;
    fn amax(n: usize, x: *mut T, incx: usize) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn gemv(
        layout: BlasLayout,
        trans: BlasTrans,
        m: usize,
        n: usize,
        alpha: T,
        a: *mut T,
        lda: usize,
        x: *mut T,
        incx: usize,
        beta: T,
        y: *mut T,
        incy: usize,
    );
    #[allow(clippy::too_many_arguments)]
    fn ger(
        layout: BlasLayout,
        m: usize,
        n: usize,
        alpha: T,
        x: *mut T,
        incx: usize,
        y: *mut T,
        incy: usize,
        a: *mut T,
        lda: usize,
    );
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        layout: BlasLayout,
        transa: BlasTrans,
        transb: BlasTrans,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: *mut T,
        lda: usize,
        b: *mut T,
        ldb: usize,
        beta: T,
        c: *mut T,
        ldc: usize,
    );
}

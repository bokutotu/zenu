use crate::num::Num;

pub enum Trans {
    N,
    T,
    C,
}

pub trait Blas {
    type Item: Num;

    fn swap(&self, n: usize, x: *mut Self::Item, incx: usize, y: *mut Self::Item, incy: usize);
    fn scal(
        &self,
        n: usize,
        alpha: Self::Item,
        x: *mut Self::Item,
        y: *mut Self::Item,
        incx: usize,
    );
    fn copy(&self, x: *mut Self::Item, y: *mut Self::Item, incx: usize, incy: usize);
    fn dot(&self, x: *mut Self::Item, y: *mut Self::Item, incx: usize, incy: usize) -> Self::Item;
    fn dotu(&self, x: *mut Self::Item, y: *mut Self::Item, incx: usize, incy: usize) -> Self::Item;
    fn norm2(&self, x: *mut Self::Item, incx: usize) -> Self::Item;
    fn asum(&self, x: *mut Self::Item, incx: usize) -> Self::Item;
    fn amax(&self, x: *mut Self::Item, incx: usize) -> usize;
    #[allow(clippy::too_many_arguments)]
    fn gemv(
        &self,
        trans: Trans,
        m: usize,
        n: usize,
        alpha: Self::Item,
        a: *mut Self::Item,
        lda: usize,
        x: *mut Self::Item,
        incx: usize,
        beta: Self::Item,
        y: *mut Self::Item,
        incy: usize,
    );
    #[allow(clippy::too_many_arguments)]
    fn ger(
        &self,
        m: usize,
        n: usize,
        alpha: Self::Item,
        x: *mut Self::Item,
        incx: usize,
        y: *mut Self::Item,
        incy: usize,
        a: *mut Self::Item,
        lda: usize,
    );
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        transa: Trans,
        transb: Trans,
        m: usize,
        n: usize,
        k: usize,
        alpha: Self::Item,
        a: *mut Self::Item,
        lda: usize,
        b: *mut Self::Item,
        ldb: usize,
        beta: Self::Item,
        c: *mut Self::Item,
        ldc: usize,
    );
}

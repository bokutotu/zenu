extern crate openblas_src;

use crate::blas::Blas;
use cblas::*;

use crate::blas::{BlasLayout, BlasTrans};

pub struct CpuBlas {}

fn from_trans(value: BlasTrans) -> Transpose {
    match value {
        BlasTrans::None => Transpose::None,
        BlasTrans::Ordinary => Transpose::Ordinary,
        BlasTrans::Conjugate => Transpose::Conjugate,
    }
}

fn from_layout(value: BlasLayout) -> Layout {
    match value {
        BlasLayout::RowMajor => Layout::RowMajor,
        BlasLayout::ColMajor => Layout::ColumnMajor,
    }
}

macro_rules! impl_blas {
    (
        $swap:ident,
        $sscal:ident,
        $copy:ident,
        $dot:ident,
        $norm2:ident,
        $asum:ident,
        $amax:ident,
        $gemv:ident,
        $ger:ident,
        $gemm:ident,
        $t:ty
    ) => {
        impl Blas<$t> for CpuBlas {
            fn swap(n: usize, x: *mut $t, incx: usize, y: *mut $t, incy: usize) {
                let x = unsafe { std::slice::from_raw_parts_mut(x, n * incx) };
                let y = unsafe { std::slice::from_raw_parts_mut(y, n * incy) };
                unsafe {
                    $swap(
                        n.try_into().unwrap(),
                        x,
                        incx.try_into().unwrap(),
                        y,
                        incy.try_into().unwrap(),
                    )
                }
            }

            fn scal(n: usize, alpha: $t, x: *mut $t, incx: usize) {
                let x = unsafe { std::slice::from_raw_parts_mut(x, n * incx) };
                unsafe { $sscal(n.try_into().unwrap(), alpha, x, incx.try_into().unwrap()) }
            }

            fn copy(n: usize, x: *const $t, incx: usize, y: *mut $t, incy: usize) {
                let x = unsafe { std::slice::from_raw_parts(x, n * incx) };
                let y = unsafe { std::slice::from_raw_parts_mut(y, n * incy) };

                let n = n.try_into().unwrap();
                let incx = incx.try_into().unwrap();
                let incy = incy.try_into().unwrap();

                unsafe { $copy(n, x, incx, y, incy) }
            }

            fn dot(n: usize, x: *mut $t, incx: usize, y: *mut $t, incy: usize) -> $t {
                let x = unsafe { std::slice::from_raw_parts(x, n * incx) };
                let y = unsafe { std::slice::from_raw_parts(y, n * incy) };

                let n = n.try_into().unwrap();
                let incx = incx.try_into().unwrap();
                let incy = incy.try_into().unwrap();

                unsafe { $dot(n, x, incx, y, incy) }
            }

            fn norm2(n: usize, x: *mut $t, incx: usize) -> $t {
                let x = unsafe { std::slice::from_raw_parts(x, n * incx) };

                let n = n.try_into().unwrap();
                let incx = incx.try_into().unwrap();

                unsafe { $norm2(n, x, incx) }
            }

            fn asum(n: usize, x: *mut $t, incx: usize) -> $t {
                let x = unsafe { std::slice::from_raw_parts(x, n * incx) };

                let n = n.try_into().unwrap();
                let incx = incx.try_into().unwrap();

                unsafe { $asum(n, x, incx) }
            }

            fn amax(n: usize, x: *mut $t, incx: usize) -> usize {
                let x = unsafe { std::slice::from_raw_parts(x, n * incx) };

                let n = n.try_into().unwrap();
                let incx = incx.try_into().unwrap();

                unsafe { $amax(n, x, incx).try_into().unwrap() }
            }

            fn gemv(
                layout: BlasLayout,
                trans: BlasTrans,
                m: usize,
                n: usize,
                alpha: $t,
                a: *mut $t,
                lda: usize,
                x: *mut $t,
                incx: usize,
                beta: $t,
                y: *mut $t,
                incy: usize,
            ) {
                let a = unsafe { std::slice::from_raw_parts(a, lda * n) };
                let x = unsafe { std::slice::from_raw_parts(x, n * incx) };
                let y = unsafe { std::slice::from_raw_parts_mut(y, m * incy) };

                let layout = from_layout(layout);
                let trans = from_trans(trans);
                let m = m.try_into().unwrap();
                let n = n.try_into().unwrap();
                let lda = lda.try_into().unwrap();
                let incx = incx.try_into().unwrap();
                let incy = incy.try_into().unwrap();

                unsafe { $gemv(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy) }
            }

            fn ger(
                layout: BlasLayout,
                m: usize,
                n: usize,
                alpha: $t,
                x: *mut $t,
                incx: usize,
                y: *mut $t,
                incy: usize,
                a: *mut $t,
                lda: usize,
            ) {
                let layout = from_layout(layout);
                let x = unsafe { std::slice::from_raw_parts(x, m * incx) };
                let y = unsafe { std::slice::from_raw_parts(y, n * incy) };
                let a = unsafe { std::slice::from_raw_parts_mut(a, lda * n) };

                let m = m.try_into().unwrap();
                let n = n.try_into().unwrap();
                let incx = incx.try_into().unwrap();
                let incy = incy.try_into().unwrap();
                let lda = lda.try_into().unwrap();

                unsafe { $ger(layout, m, n, alpha, x, incx, y, incy, a, lda) }
            }

            fn gemm(
                layout: BlasLayout,
                transa: BlasTrans,
                transb: BlasTrans,
                m: usize,
                n: usize,
                k: usize,
                alpha: $t,
                a: *mut $t,
                lda: usize,
                b: *mut $t,
                ldb: usize,
                beta: $t,
                c: *mut $t,
                ldc: usize,
            ) {
                let layout = from_layout(layout);
                let transa = from_trans(transa);
                let transb = from_trans(transb);

                let a = unsafe { std::slice::from_raw_parts(a, lda) };
                let b = unsafe { std::slice::from_raw_parts(b, ldb) };
                let c = unsafe { std::slice::from_raw_parts_mut(c, ldc) };

                let m = m.try_into().unwrap();
                let n = n.try_into().unwrap();
                let k = k.try_into().unwrap();
                let lda = lda.try_into().unwrap();
                let ldb = ldb.try_into().unwrap();
                let ldc = ldc.try_into().unwrap();

                unsafe {
                    $gemm(
                        layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                    )
                }
            }
        }
    };
}

impl_blas!(sswap, sscal, scopy, sdot, snrm2, sasum, isamax, sgemv, sger, sgemm, f32);
impl_blas!(dswap, dscal, dcopy, ddot, dnrm2, dasum, idamax, dgemv, dger, dgemm, f64);

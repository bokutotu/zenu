extern crate openblas_src;

use std::marker::PhantomData;

use crate::{blas::Blas, num::Num};
use cblas::*;

use crate::blas::{BlasLayout, BlasTrans};

pub struct CpuBlas<T: Num> {
    _phantom: PhantomData<T>,
}

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

impl<N: Num> Blas<N> for CpuBlas<N> {
    fn swap(n: usize, x: *mut N, incx: usize, y: *mut N, incy: usize) {
        if N::is_f32() {
            let x = unsafe { std::slice::from_raw_parts_mut(x as *mut f32, n * incx) };
            let y = unsafe { std::slice::from_raw_parts_mut(y as *mut f32, n * incy) };
            unsafe {
                sswap(
                    n.try_into().unwrap(),
                    x,
                    incx.try_into().unwrap(),
                    y,
                    incy.try_into().unwrap(),
                )
            }
        } else {
            let x = unsafe { std::slice::from_raw_parts_mut(x as *mut f64, n * incx) };
            let y = unsafe { std::slice::from_raw_parts_mut(y as *mut f64, n * incy) };
            unsafe {
                dswap(
                    n.try_into().unwrap(),
                    x,
                    incx.try_into().unwrap(),
                    y,
                    incy.try_into().unwrap(),
                )
            }
        }
    }

    fn scal(n: usize, alpha: N, x: *mut N, incx: usize) {
        if N::is_f32() {
            let x = unsafe { std::slice::from_raw_parts_mut(x as *mut f32, n * incx) };
            unsafe {
                sscal(
                    n.try_into().unwrap(),
                    *(&alpha as *const N as *const f32),
                    x,
                    incx.try_into().unwrap(),
                )
            }
        } else {
            let x = unsafe { std::slice::from_raw_parts_mut(x as *mut f64, n * incx) };
            unsafe {
                dscal(
                    n.try_into().unwrap(),
                    *(&alpha as *const N as *const f64),
                    x,
                    incx.try_into().unwrap(),
                )
            }
        }
    }

    fn axpy(n: usize, alpha: N, x: *const N, incx: usize, y: *mut N, incy: usize) {
        if N::is_f32() {
            let x = unsafe { std::slice::from_raw_parts(x as *mut f32, 1) };
            let y = unsafe { std::slice::from_raw_parts_mut(y as *mut f32, 1) };
            unsafe {
                saxpy(
                    n.try_into().unwrap(),
                    *(&alpha as *const N as *const f32),
                    x,
                    incx.try_into().unwrap(),
                    y,
                    incy.try_into().unwrap(),
                )
            }
        } else {
            let x = unsafe { std::slice::from_raw_parts(x as *mut f64, 1) };
            let y = unsafe { std::slice::from_raw_parts_mut(y as *mut f64, 1) };
            unsafe {
                daxpy(
                    n.try_into().unwrap(),
                    *(&alpha as *const N as *const f64),
                    x,
                    incx.try_into().unwrap(),
                    y,
                    incy.try_into().unwrap(),
                )
            }
        }
    }

    fn copy(n: usize, x: *const N, incx: usize, y: *mut N, incy: usize) {
        if N::is_f32() {
            let x = unsafe { std::slice::from_raw_parts(x as *const f32, n * incx) };
            let y = unsafe { std::slice::from_raw_parts_mut(y as *mut f32, n * incy) };
            unsafe {
                scopy(
                    n.try_into().unwrap(),
                    x,
                    incx.try_into().unwrap(),
                    y,
                    incy.try_into().unwrap(),
                )
            }
        } else {
            let x = unsafe { std::slice::from_raw_parts(x as *const f64, n * incx) };
            let y = unsafe { std::slice::from_raw_parts_mut(y as *mut f64, n * incy) };
            unsafe {
                dcopy(
                    n.try_into().unwrap(),
                    x,
                    incx.try_into().unwrap(),
                    y,
                    incy.try_into().unwrap(),
                )
            }
        }
    }

    fn dot(n: usize, x: *const N, incx: usize, y: *const N, incy: usize) -> N {
        if N::is_f32() {
            let x = unsafe { std::slice::from_raw_parts(x as *const f32, n * incx) };
            let y = unsafe { std::slice::from_raw_parts(y as *const f32, n * incy) };
            unsafe {
                *(&sdot(
                    n.try_into().unwrap(),
                    x,
                    incx.try_into().unwrap(),
                    y,
                    incy.try_into().unwrap(),
                ) as *const f32 as *const N)
            }
        } else {
            let x = unsafe { std::slice::from_raw_parts(x as *const f64, n * incx) };
            let y = unsafe { std::slice::from_raw_parts(y as *const f64, n * incy) };
            unsafe {
                *(&ddot(
                    n.try_into().unwrap(),
                    x,
                    incx.try_into().unwrap(),
                    y,
                    incy.try_into().unwrap(),
                ) as *const f64 as *const N)
            }
        }
    }

    fn norm2(n: usize, x: *mut N, incx: usize) -> N {
        if N::is_f32() {
            let x = unsafe { std::slice::from_raw_parts(x as *const N as *const f32, n * incx) };
            unsafe {
                *(&snrm2(n.try_into().unwrap(), x, incx.try_into().unwrap()) as *const f32
                    as *const N)
            }
        } else {
            let x = unsafe { std::slice::from_raw_parts(x as *const N as *const f64, n * incx) };
            unsafe {
                *(&dnrm2(n.try_into().unwrap(), x, incx.try_into().unwrap()) as *const f64
                    as *const N)
            }
        }
    }

    fn asum(n: usize, x: *const N, incx: usize) -> N {
        if N::is_f32() {
            let x = unsafe { std::slice::from_raw_parts(x as *const f32, n * incx) };
            unsafe {
                *(&sasum(n.try_into().unwrap(), x, incx.try_into().unwrap()) as *const f32
                    as *const N)
            }
        } else {
            let x = unsafe { std::slice::from_raw_parts(x as *const f64, n * incx) };
            unsafe {
                *(&dasum(n.try_into().unwrap(), x, incx.try_into().unwrap()) as *const f64
                    as *const N)
            }
        }
    }

    fn amax(n: usize, x: *const N, incx: usize) -> usize {
        if N::is_f32() {
            let x = unsafe { std::slice::from_raw_parts(x as *const f32, n * incx) };
            unsafe { isamax(n.try_into().unwrap(), x, incx.try_into().unwrap()) }
                .try_into()
                .unwrap()
        } else {
            let x = unsafe { std::slice::from_raw_parts(x as *const f64, n * incx) };
            unsafe { idamax(n.try_into().unwrap(), x, incx.try_into().unwrap()) }
                .try_into()
                .unwrap()
        }
    }

    fn gemv(
        layout: BlasLayout,
        trans: BlasTrans,
        m: usize,
        n: usize,
        alpha: N,
        a: *mut N,
        lda: usize,
        x: *mut N,
        incx: usize,
        beta: N,
        y: *mut N,
        incy: usize,
    ) {
        if N::is_f32() {
            let a = unsafe { std::slice::from_raw_parts(a as *const f32, lda * n) };
            let x = unsafe { std::slice::from_raw_parts(x as *const f32, n * incx) };
            let y = unsafe { std::slice::from_raw_parts_mut(y as *mut f32, m * incy) };

            let layout = from_layout(layout);
            let trans = from_trans(trans);
            let m = m.try_into().unwrap();
            let n = n.try_into().unwrap();
            let lda = lda.try_into().unwrap();
            let incx = incx.try_into().unwrap();
            let incy = incy.try_into().unwrap();

            unsafe {
                sgemv(
                    layout,
                    trans,
                    m,
                    n,
                    *(&alpha as *const N as *const f32),
                    a,
                    lda,
                    x,
                    incx,
                    *(&beta as *const N as *const f32),
                    y,
                    incy,
                )
            }
        } else {
            let a = unsafe { std::slice::from_raw_parts(a as *const f64, lda * n) };
            let x = unsafe { std::slice::from_raw_parts(x as *const f64, n * incx) };
            let y = unsafe { std::slice::from_raw_parts_mut(y as *mut f64, m * incy) };

            let layout = from_layout(layout);
            let trans = from_trans(trans);
            let m = m.try_into().unwrap();
            let n = n.try_into().unwrap();
            let lda = lda.try_into().unwrap();
            let incx = incx.try_into().unwrap();
            let incy = incy.try_into().unwrap();

            unsafe {
                dgemv(
                    layout,
                    trans,
                    m,
                    n,
                    *(&alpha as *const N as *const f64),
                    a,
                    lda,
                    x,
                    incx,
                    *(&beta as *const N as *const f64),
                    y,
                    incy,
                )
            }
        }
    }

    fn ger(
        layout: BlasLayout,
        m: usize,
        n: usize,
        alpha: N,
        x: *mut N,
        incx: usize,
        y: *mut N,
        incy: usize,
        a: *mut N,
        lda: usize,
    ) {
        let layout = from_layout(layout);

        if N::is_f32() {
            let x = unsafe { std::slice::from_raw_parts(x as *const f32, m * incx) };
            let y = unsafe { std::slice::from_raw_parts(y as *const f32, n * incy) };
            let a = unsafe { std::slice::from_raw_parts_mut(a as *mut f32, lda * n) };

            let m = m.try_into().unwrap();
            let n = n.try_into().unwrap();
            let incx = incx.try_into().unwrap();
            let incy = incy.try_into().unwrap();
            let lda = lda.try_into().unwrap();

            unsafe {
                sger(
                    layout,
                    m,
                    n,
                    *(&alpha as *const N as *const f32),
                    x,
                    incx,
                    y,
                    incy,
                    a,
                    lda,
                )
            }
        } else {
            let x = unsafe { std::slice::from_raw_parts(x as *const f64, m * incx) };
            let y = unsafe { std::slice::from_raw_parts(y as *const f64, n * incy) };
            let a = unsafe { std::slice::from_raw_parts_mut(a as *mut f64, lda * n) };

            let m = m.try_into().unwrap();
            let n = n.try_into().unwrap();
            let incx = incx.try_into().unwrap();
            let incy = incy.try_into().unwrap();
            let lda = lda.try_into().unwrap();

            unsafe {
                dger(
                    layout,
                    m,
                    n,
                    *(&alpha as *const N as *const f64),
                    x,
                    incx,
                    y,
                    incy,
                    a,
                    lda,
                )
            }
        }
    }

    fn gemm(
        layout: BlasLayout,
        transa: BlasTrans,
        transb: BlasTrans,
        m: usize,
        n: usize,
        k: usize,
        alpha: N,
        a: *mut N,
        lda: usize,
        b: *mut N,
        ldb: usize,
        beta: N,
        c: *mut N,
        ldc: usize,
    ) {
        let layout = from_layout(layout);
        let transa = from_trans(transa);
        let transb = from_trans(transb);

        if N::is_f32() {
            let a = unsafe { std::slice::from_raw_parts(a as *const f32, lda) };
            let b = unsafe { std::slice::from_raw_parts(b as *const f32, ldb) };
            let c = unsafe { std::slice::from_raw_parts_mut(c as *mut f32, ldc) };

            let m = m.try_into().unwrap();
            let n = n.try_into().unwrap();
            let k = k.try_into().unwrap();
            let lda = lda.try_into().unwrap();
            let ldb = ldb.try_into().unwrap();
            let ldc = ldc.try_into().unwrap();

            unsafe {
                sgemm(
                    layout,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    *(&alpha as *const N as *const f32),
                    a,
                    lda,
                    b,
                    ldb,
                    *(&beta as *const N as *const f32),
                    c,
                    ldc,
                )
            }
        } else {
            let a = unsafe { std::slice::from_raw_parts(a as *const f64, lda) };
            let b = unsafe { std::slice::from_raw_parts(b as *const f64, ldb) };
            let c = unsafe { std::slice::from_raw_parts_mut(c as *mut f64, ldc) };

            let m = m.try_into().unwrap();
            let n = n.try_into().unwrap();
            let k = k.try_into().unwrap();
            let lda = lda.try_into().unwrap();
            let ldb = ldb.try_into().unwrap();
            let ldc = ldc.try_into().unwrap();

            unsafe {
                dgemm(
                    layout,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    *(&alpha as *const N as *const f64),
                    a,
                    lda,
                    b,
                    ldb,
                    *(&beta as *const N as *const f64),
                    c,
                    ldc,
                )
            }
        }
    }
}

#[cfg(test)]
mod cpu_blas {
    use super::*;
    fn zero_vec(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    fn fill_range(n: usize) -> Vec<f32> {
        (0..n).map(|x| x as f32).collect()
    }

    fn zero_vec_f64(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }

    fn fill_range_f64(n: usize) -> Vec<f64> {
        (0..n).map(|x| x as f64).collect()
    }

    #[test]
    fn f32_swap() {
        let mut x = zero_vec(10);
        let mut y = fill_range(10);

        super::CpuBlas::<f32>::swap(10, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1);

        assert_eq!(x, fill_range(10));
        assert_eq!(y, zero_vec(10));
    }

    #[test]
    fn f64_swap() {
        let mut x = zero_vec_f64(10);
        let mut y = fill_range_f64(10);

        super::CpuBlas::<f64>::swap(10, x.as_mut_ptr(), 1, y.as_mut_ptr(), 1);

        assert_eq!(x, fill_range_f64(10));
        assert_eq!(y, zero_vec_f64(10));
    }
}

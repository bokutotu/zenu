use crate::{
    device::{cpu::Cpu, Device},
    dim::DimTrait,
    matrix::{Matrix, Ref},
    num::Num,
};

pub trait Copy: Device {
    fn copy_raw<T: Num>(n: usize, x: *const T, incx: usize, y: *mut T, incy: usize);
}

impl Copy for Cpu {
    fn copy_raw<T: Num>(n: usize, x: *const T, incx: usize, y: *mut T, incy: usize) {
        extern crate openblas_src;
        use cblas::*;
        if T::is_f32() {
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
}

pub fn copy_unchecked<T, SA, SB, RB, D>(x: Matrix<Ref<&T>, SA, D>, y: Matrix<Ref<&mut T>, SB, D>)
where
    T: Num,
    SA: DimTrait,
    SB: DimTrait,
    D: Copy,
{
    let n = x.shape()[0];
    let incx = x.stride()[0];
    let incy = y.stride()[0];
    let x = x.as_ptr();
    let y = y.as_mut_ptr();
    D::copy_raw(n, x, incx, y, incy);
}

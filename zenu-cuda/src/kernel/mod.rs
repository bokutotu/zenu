use std::any::TypeId;

use zenu_cuda_kernel_sys::*;

macro_rules! impl_array_scalar {
    ($name:ident, $double_fn:ident, $float_fn:ident) => {
        pub fn $name<T: 'static>(a: *mut T, size: usize, stride: usize, scalar: T, out: *mut T) {
            let size = size as ::std::os::raw::c_int;
            let stride = stride as ::std::os::raw::c_int;
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a as *mut f32;
                let out = out as *mut f32;
                let scalar = unsafe { *{ &scalar as *const T as *const f32 } };
                unsafe { $float_fn(a, size, stride, scalar, out) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a as *mut f64;
                let out = out as *mut f64;
                let scalar = unsafe { *{ &scalar as *const T as *const f64 } };
                unsafe { $double_fn(a, size, stride, scalar, out) }
            }
        }
    };
}
impl_array_scalar!(
    array_scalar_add,
    array_scalar_add_double,
    array_scalar_add_float
);
impl_array_scalar!(
    array_scalar_sub,
    array_scalar_sub_double,
    array_scalar_sub_float
);
impl_array_scalar!(
    array_scalar_mul,
    array_scalar_mul_double,
    array_scalar_mul_float
);
impl_array_scalar!(
    array_scalar_div,
    array_scalar_div_double,
    array_scalar_div_float
);

#[cfg(test)]
mod array_scalar {
    use crate::runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind};

    use super::*;

    macro_rules! impl_array_scalr_test {
        ($test_name:ident, $input:expr, $scalar:expr, $ans:expr, $ty:ty, $kernel_func:ident) => {
            #[test]
            fn $test_name() {
                let a: Vec<$ty> = $input;
                let mut out = vec![0 as $ty; a.len()];
                let scalar: $ty = $scalar;
                let a_gpu = cuda_malloc(a.len()).unwrap();
                cuda_copy(
                    a_gpu,
                    a.as_ptr(),
                    a.len(),
                    ZenuCudaMemCopyKind::HostToDevice,
                )
                .unwrap();
                let out_gpu = cuda_malloc(out.len()).unwrap();
                $kernel_func(a_gpu, a.len(), 1, scalar, out_gpu);
                cuda_copy(
                    out.as_mut_ptr(),
                    out_gpu,
                    out.len(),
                    ZenuCudaMemCopyKind::DeviceToHost,
                )
                .unwrap();
                let ans: Vec<$ty> = $ans;
                assert_eq!(out, ans);
            }
        };
    }
    impl_array_scalr_test!(
        add_scalar_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        2.,
        vec![3.0, 4.0, 5.0, 6.0],
        f32,
        array_scalar_add
    );

    impl_array_scalr_test!(
        add_scalar_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        2.,
        vec![3.0, 4.0, 5.0, 6.0],
        f64,
        array_scalar_add
    );

    impl_array_scalr_test!(
        sub_scalar_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        2.,
        vec![-1.0, 0.0, 1.0, 2.0],
        f32,
        array_scalar_sub
    );

    impl_array_scalr_test!(
        sub_scalar_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        2.,
        vec![-1.0, 0.0, 1.0, 2.0],
        f64,
        array_scalar_sub
    );

    impl_array_scalr_test!(
        mul_scalar_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        2.,
        vec![2.0, 4.0, 6.0, 8.0],
        f32,
        array_scalar_mul
    );

    impl_array_scalr_test!(
        mul_scalar_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        2.,
        vec![2.0, 4.0, 6.0, 8.0],
        f64,
        array_scalar_mul
    );

    impl_array_scalr_test!(
        div_scalar_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        2.,
        vec![0.5, 1.0, 1.5, 2.0],
        f32,
        array_scalar_div
    );

    impl_array_scalr_test!(
        div_scalar_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        2.,
        vec![0.5, 1.0, 1.5, 2.0],
        f64,
        array_scalar_div
    );
}

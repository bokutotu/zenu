use std::any::TypeId;

use zenu_cuda_kernel_sys::*;

macro_rules! impl_array_scalar {
    ($name:ident, $double_fn:ident, $float_fn:ident) => {
        pub fn $name<T: 'static>(
            out: *mut T,
            a: *const T,
            scalar: T,
            size: usize,
            out_stride: usize,
            stride: usize,
        ) {
            let size = size as ::std::os::raw::c_int;
            let stride = stride as ::std::os::raw::c_int;
            let out_stride = out_stride as i32;
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a as *mut f32;
                let out = out as *mut f32;
                let scalar = unsafe { *{ &scalar as *const T as *const f32 } };
                unsafe { $float_fn(a, size, stride, scalar, out, out_stride) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a as *mut f64;
                let out = out as *mut f64;
                let scalar = unsafe { *{ &scalar as *const T as *const f64 } };
                unsafe { $double_fn(a, size, stride, scalar, out, out_stride) }
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

macro_rules! impl_array_scalar_assign {
    ($name:ident, $double_fn:ident, $float_fn:ident) => {
        pub fn $name<T: 'static>(a: *mut T, scalar: T, size: usize, stride: usize) {
            let size = size as ::std::os::raw::c_int;
            let stride = stride as ::std::os::raw::c_int;
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a as *mut f32;
                let scalar = unsafe { *{ &scalar as *const T as *const f32 } };
                unsafe { $float_fn(a, size, stride, scalar) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a as *mut f64;
                let scalar = unsafe { *{ &scalar as *const T as *const f64 } };
                unsafe { $double_fn(a, size, stride, scalar) }
            }
        }
    };
}
impl_array_scalar_assign!(
    array_scalar_add_assign,
    array_scalar_add_assign_double,
    array_scalar_add_assign_float
);
impl_array_scalar_assign!(
    array_scalar_sub_assign,
    array_scalar_sub_assign_double,
    array_scalar_sub_assign_float
);
impl_array_scalar_assign!(
    array_scalar_mul_assign,
    array_scalar_mul_assign_double,
    array_scalar_mul_assign_float
);
impl_array_scalar_assign!(
    array_scalar_div_assign,
    array_scalar_div_assign_double,
    array_scalar_div_assign_float
);

macro_rules! impl_arra_array {
    ($name:ident, $double_fn:ident, $float_fn:ident) => {
        pub fn $name<T: 'static>(
            c: *mut T,
            a: *const T,
            b: *const T,
            size: usize,
            stride_c: usize,
            stride_a: usize,
            stride_b: usize,
        ) {
            let size = size as ::std::os::raw::c_int;
            let stride_a = stride_a as ::std::os::raw::c_int;
            let stride_b = stride_b as ::std::os::raw::c_int;
            let stride_c = stride_c as ::std::os::raw::c_int;
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a as *mut f32;
                let b = b as *mut f32;
                let c = c as *mut f32;
                unsafe { $float_fn(a, stride_a, b, stride_b, c, stride_c, size) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a as *mut f64;
                let b = b as *mut f64;
                let c = c as *mut f64;
                unsafe { $double_fn(a, stride_a, b, stride_b, c, stride_c, size) }
            }
        }
    };
}
impl_arra_array!(array_add, array_array_add_double, array_array_add_float);
impl_arra_array!(array_sub, array_array_sub_double, array_array_sub_float);
impl_arra_array!(array_mul, array_array_mul_double, array_array_mul_float);
impl_arra_array!(array_div, array_array_div_double, array_array_div_float);

macro_rules! impl_array_array_assign {
    ($name:ident, $double_fn:ident, $float_fn:ident) => {
        pub fn $name<T: 'static>(
            a: *mut T,
            b: *const T,
            size: usize,
            stride_a: usize,
            stride_b: usize,
        ) {
            let size = size as ::std::os::raw::c_int;
            let stride_a = stride_a as ::std::os::raw::c_int;
            let stride_b = stride_b as ::std::os::raw::c_int;
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a as *mut f32;
                let b = b as *mut f32;
                unsafe { $float_fn(a, stride_a, b, stride_b, size) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a as *mut f64;
                let b = b as *mut f64;
                unsafe { $double_fn(a, stride_a, b, stride_b, size) }
            }
        }
    };
}
impl_array_array_assign!(
    array_array_add_assign,
    array_array_add_assign_double,
    array_array_add_assign_float
);
impl_array_array_assign!(
    array_array_sub_assign,
    array_array_sub_assign_double,
    array_array_sub_assign_float
);
impl_array_array_assign!(
    array_array_mul_assign,
    array_array_mul_assign_double,
    array_array_mul_assign_float
);
impl_array_array_assign!(
    array_array_div_assign,
    array_array_div_assign_double,
    array_array_div_assign_float
);

macro_rules! impl_array_scalar_sin {
    ($name:ident, $double_fn:ident, $float_fn:ident) => {
        pub fn $name<T: 'static>(a: *mut T, size: usize, stride: usize, out: *mut T) {
            let size = size as ::std::os::raw::c_int;
            let stride = stride as ::std::os::raw::c_int;
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a as *mut f32;
                let out = out as *mut f32;
                unsafe { $float_fn(a, size, stride, out) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a as *mut f64;
                let out = out as *mut f64;
                unsafe { $double_fn(a, size, stride, out) }
            }
        }
    };
}
impl_array_scalar_sin!(array_sin, array_sin_double, array_sin_float);
impl_array_scalar_sin!(array_cos, array_cos_double, array_cos_float);
impl_array_scalar_sin!(array_tan, array_tan_double, array_tan_float);
impl_array_scalar_sin!(array_asin, array_asin_double, array_asin_float);
impl_array_scalar_sin!(array_acos, array_acos_double, array_acos_float);
impl_array_scalar_sin!(array_atan, array_atan_double, array_atan_float);
impl_array_scalar_sin!(array_sinh, array_sinh_double, array_sinh_float);
impl_array_scalar_sin!(array_cosh, array_cosh_double, array_cosh_float);
impl_array_scalar_sin!(array_tanh, array_tanh_double, array_tanh_float);
impl_array_scalar_sin!(array_abs, array_abs_double, array_abs_float);
impl_array_scalar_sin!(array_sqrt, array_sqrt_double, array_sqrt_float);
impl_array_scalar_sin!(array_exp, array_exp_double, array_exp_float);

macro_rules! impl_array_scalar_sin_assign {
    ($name:ident, $double_fn:ident, $float_fn:ident) => {
        pub fn $name<T: 'static>(a: *mut T, size: usize, stride: usize) {
            let size = size as ::std::os::raw::c_int;
            let stride = stride as ::std::os::raw::c_int;
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a as *mut f32;
                unsafe { $float_fn(a, size, stride) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a as *mut f64;
                unsafe { $double_fn(a, size, stride) }
            }
        }
    };
}
impl_array_scalar_sin_assign!(
    array_sin_assign,
    array_sin_assign_double,
    array_sin_assign_float
);
impl_array_scalar_sin_assign!(
    array_cos_assign,
    array_cos_assign_double,
    array_cos_assign_float
);
impl_array_scalar_sin_assign!(
    array_tan_assign,
    array_tan_assign_double,
    array_tan_assign_float
);
impl_array_scalar_sin_assign!(
    array_asin_assign,
    array_asin_assign_double,
    array_asin_assign_float
);
impl_array_scalar_sin_assign!(
    array_acos_assign,
    array_acos_assign_double,
    array_acos_assign_float
);
impl_array_scalar_sin_assign!(
    array_atan_assign,
    array_atan_assign_double,
    array_atan_assign_float
);
impl_array_scalar_sin_assign!(
    array_sinh_assign,
    array_sinh_assign_double,
    array_sinh_assign_float
);
impl_array_scalar_sin_assign!(
    array_cosh_assign,
    array_cosh_assign_double,
    array_cosh_assign_float
);
impl_array_scalar_sin_assign!(
    array_tanh_assign,
    array_tanh_assign_double,
    array_tanh_assign_float
);
impl_array_scalar_sin_assign!(
    array_abs_assign,
    array_abs_assign_double,
    array_abs_assign_float
);
impl_array_scalar_sin_assign!(
    array_sqrt_assign,
    array_sqrt_assign_double,
    array_sqrt_assign_float
);
impl_array_scalar_sin_assign!(
    array_exp_assign,
    array_exp_assign_double,
    array_exp_assign_float
);

pub fn get_memory<T: 'static + Default>(array: *const T, offset: usize) -> T {
    let mut out: T = Default::default();
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let array = array as *mut f32;
        unsafe {
            memory_access_float(array, offset as libc::c_int, &mut out as *mut T as *mut f32)
        };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let array = array as *mut f64;
        unsafe {
            memory_access_double(array, offset as libc::c_int, &mut out as *mut T as *mut f64)
        };
    }
    out
}

pub fn set_memory<T: 'static>(array: *mut T, offset: usize, value: T) {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let array = array as *mut f32;
        let value = unsafe { *{ &value as *const T as *const f32 } };
        unsafe { memory_set_float(array, offset as libc::c_int, value) };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let array = array as *mut f64;
        let value = unsafe { *{ &value as *const T as *const f64 } };
        unsafe { memory_set_double(array, offset as libc::c_int, value) };
    }
}

pub fn clip<T: 'static>(
    input: *const T,
    output: *mut T,
    size: usize,
    stride_in: usize,
    stride_out: usize,
    min: T,
    max: T,
) {
    let size = size as ::std::os::raw::c_int;
    let stride_in = stride_in as ::std::os::raw::c_int;
    let stride_out = stride_out as ::std::os::raw::c_int;
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let input = input as *mut f32;
        let output = output as *mut f32;
        let min = unsafe { *{ &min as *const T as *const f32 } };
        let max = unsafe { *{ &max as *const T as *const f32 } };
        unsafe { array_clip_float(input, output, size, stride_in, stride_out, min, max) };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let input = input as *mut f64;
        let output = output as *mut f64;
        let min = unsafe { *{ &min as *const T as *const f64 } };
        let max = unsafe { *{ &max as *const T as *const f64 } };
        unsafe { array_clip_double(input, output, size, stride_in, stride_out, min, max) };
    }
}

pub fn clip_assign<T: 'static>(input: *mut T, size: usize, stride: usize, min: T, max: T) {
    let size = size as ::std::os::raw::c_int;
    let stride = stride as ::std::os::raw::c_int;
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let input = input as *mut f32;
        let min = unsafe { *{ &min as *const T as *const f32 } };
        let max = unsafe { *{ &max as *const T as *const f32 } };
        unsafe { array_clip_assign_float(input, size, stride, min, max) };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let input = input as *mut f64;
        let min = unsafe { *{ &min as *const T as *const f64 } };
        let max = unsafe { *{ &max as *const T as *const f64 } };
        unsafe { array_clip_assign_double(input, size, stride, min, max) };
    }
}

#[cfg(test)]
mod array_array {
    use super::*;
    use crate::runtime::*;

    macro_rules! impl_array_array_test {
        ($test_name:ident, $input_1:expr, $input_2:expr, $ans:expr, $ty:ty, $kernel_func:ident) => {
            #[test]
            fn $test_name() {
                let a: Vec<$ty> = $input_1;
                let b: Vec<$ty> = $input_2;
                let mut out = vec![0 as $ty; a.len()];
                let a_gpu = cuda_malloc(a.len()).unwrap();
                let b_gpu = cuda_malloc(b.len()).unwrap();
                let out_gpu = cuda_malloc(out.len()).unwrap();
                cuda_copy(
                    a_gpu,
                    a.as_ptr(),
                    a.len(),
                    ZenuCudaMemCopyKind::HostToDevice,
                )
                .unwrap();
                cuda_copy(
                    b_gpu,
                    b.as_ptr(),
                    b.len(),
                    ZenuCudaMemCopyKind::HostToDevice,
                )
                .unwrap();
                // $kernel_func(a_gpu, 1, b_gpu, 1, out_gpu, 1, a.len());
                $kernel_func(out_gpu, a_gpu, b_gpu, a.len(), 1, 1, 1);
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
    impl_array_array_test!(
        add_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 4.0, 6.0, 8.0],
        f32,
        array_add
    );
    impl_array_array_test!(
        add_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 4.0, 6.0, 8.0],
        f64,
        array_add
    );
    impl_array_array_test!(
        sub_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, 0.0, 0.0, 0.0],
        f32,
        array_sub
    );
    impl_array_array_test!(
        sub_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, 0.0, 0.0, 0.0],
        f64,
        array_sub
    );
    impl_array_array_test!(
        mul_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 4.0, 9.0, 16.0],
        f32,
        array_mul
    );
    impl_array_array_test!(
        mul_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 4.0, 9.0, 16.0],
        f64,
        array_mul
    );
    impl_array_array_test!(
        div_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 1.0, 1.0, 1.0],
        f32,
        array_div
    );
    impl_array_array_test!(
        div_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 1.0, 1.0, 1.0],
        f64,
        array_div
    );
    macro_rules! impl_array_array_assign_test {
        ($name:ident, $input:expr, $input2:expr, $ans:expr, $ty:ty, $kernel_func:ident) => {
            #[test]
            fn $name() {
                let a: Vec<$ty> = $input;
                let b: Vec<$ty> = $input2;
                let mut out = vec![0 as $ty; a.len()];
                let a_gpu = cuda_malloc(a.len()).unwrap();
                cuda_copy(
                    a_gpu,
                    a.as_ptr(),
                    a.len(),
                    ZenuCudaMemCopyKind::HostToDevice,
                )
                .unwrap();
                let b_gpu = cuda_malloc(b.len()).unwrap();
                cuda_copy(
                    b_gpu,
                    b.as_ptr(),
                    b.len(),
                    ZenuCudaMemCopyKind::HostToDevice,
                )
                .unwrap();
                // $kernel_func(a_gpu, 1, b_gpu, 1, a.len());
                $kernel_func(a_gpu, b_gpu, a.len(), 1, 1);
                cuda_copy(
                    out.as_mut_ptr(),
                    a_gpu,
                    a.len(),
                    ZenuCudaMemCopyKind::DeviceToHost,
                )
                .unwrap();
                let ans: Vec<$ty> = $ans;
                assert_eq!(out, ans);
            }
        };
    }
    impl_array_array_assign_test!(
        add_assign_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 4.0, 6.0, 8.0],
        f32,
        array_array_add_assign
    );
    impl_array_array_assign_test!(
        add_assign_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 4.0, 6.0, 8.0],
        f64,
        array_array_add_assign
    );
    impl_array_array_assign_test!(
        sub_assign_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, 0.0, 0.0, 0.0],
        f32,
        array_array_sub_assign
    );
    impl_array_array_assign_test!(
        sub_assign_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, 0.0, 0.0, 0.0],
        f64,
        array_array_sub_assign
    );
    impl_array_array_assign_test!(
        mul_assign_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 4.0, 9.0, 16.0],
        f32,
        array_array_mul_assign
    );
    impl_array_array_assign_test!(
        mul_assign_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 4.0, 9.0, 16.0],
        f64,
        array_array_mul_assign
    );
    impl_array_array_assign_test!(
        div_assign_f32,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 1.0, 1.0, 1.0],
        f32,
        array_array_div_assign
    );
    impl_array_array_assign_test!(
        div_assign_f64,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 1.0, 1.0, 1.0],
        f64,
        array_array_div_assign
    );
}

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
                // $kernel_func(a_gpu, a.len(), 1, scalar, out_gpu, 1);
                $kernel_func(out_gpu, a_gpu, scalar, a.len(), 1, 1);
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

    macro_rules! impl_test_sin {
        ($name:ident, $input:expr, $ans:expr, $ty:ty, $func:ident) => {
            #[test]
            fn $name() {
                let a: Vec<$ty> = $input;
                let mut out = vec![0.0; a.len()];
                let a_gpu = cuda_malloc(a.len()).unwrap();
                cuda_copy(
                    a_gpu,
                    a.as_ptr(),
                    a.len(),
                    ZenuCudaMemCopyKind::HostToDevice,
                )
                .unwrap();
                let out_gpu = cuda_malloc(out.len()).unwrap();
                $func(a_gpu, a.len(), 1, out_gpu);
                cuda_copy(
                    out.as_mut_ptr(),
                    out_gpu,
                    out.len(),
                    ZenuCudaMemCopyKind::DeviceToHost,
                )
                .unwrap();
                let ans: Vec<$ty> = $ans;
                for (a, b) in out.iter().zip(ans.iter()) {
                    assert!((a - b).abs() < 1e-6);
                }
            }
        };
    }
    impl_test_sin!(
        sin_f32,
        vec![0.0, 1.0, 2.0, 3.0],
        vec![0.0, 0.84147096, 0.9092974, 0.14112001],
        f32,
        array_sin
    );
    impl_test_sin!(
        sin_f64,
        vec![0.0, 1.0, 2.0, 3.0],
        vec![
            0.0,
            0.8414709848078965,
            0.9092974268256817,
            0.1411200080598672
        ],
        f64,
        array_sin
    );
    impl_test_sin!(
        cos_f32,
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 0.5403023, -0.41614684, -0.9899925],
        f32,
        array_cos
    );
    impl_test_sin!(
        cos_f64,
        vec![0.0, 1.0, 2.0, 3.0],
        vec![
            1.0,
            0.5403023058681398,
            -0.4161468365471424,
            -0.9899924966004454
        ],
        f64,
        array_cos
    );
    impl_test_sin!(
        tan_f32,
        vec![0.0, 1.0, 2.0, 3.0],
        vec![0.0, 1.5574077, -2.1850398, -0.14254654],
        f32,
        array_tan
    );
    impl_test_sin!(
        tan_f64,
        vec![0.0, 1.0, 2.0, 3.0],
        vec![
            0.0,
            1.5574077246549023,
            -2.185039863261519,
            -0.1425465430742778
        ],
        f64,
        array_tan
    );
    impl_test_sin!(
        asin_f32,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![0.0, 0.5235988, 0.9272952, 1.5707964],
        f32,
        array_asin
    );
    impl_test_sin!(
        asin_f64,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![
            0.0,
            0.5235987755982989,
            0.9272952180016122,
            1.5707963267948966
        ],
        f64,
        array_asin
    );
    impl_test_sin!(
        acos_f32,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![1.5707964, 1.0471976, 0.6435011, 0.0],
        f32,
        array_acos
    );
    impl_test_sin!(
        acos_f64,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![
            1.5707963267948966,
            1.0471975511965979,
            0.6435011087932844,
            0.0
        ],
        f64,
        array_acos
    );
    impl_test_sin!(
        atan_f32,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![0.0, 0.4636476, 0.6747409, 0.7853982],
        f32,
        array_atan
    );
    impl_test_sin!(
        atan_f64,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![
            0.0,
            0.4636476090008061,
            0.6747409422235527,
            0.7853981633974483
        ],
        f64,
        array_atan
    );
    impl_test_sin!(
        sinh_f32,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![0.0, 0.5210953, 0.88810597, 1.1752012],
        f32,
        array_sinh
    );
    impl_test_sin!(
        sinh_f64,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![
            0.0,
            0.5210953054937474,
            0.888105982187623,
            1.1752011936438014
        ],
        f64,
        array_sinh
    );
    impl_test_sin!(
        cosh_f32,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![1., 1.12762597, 1.33743495, 1.54308063],
        f32,
        array_cosh
    );
    impl_test_sin!(
        cosh_f64,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![1., 1.12762597, 1.33743495, 1.54308063],
        f64,
        array_cosh
    );
    impl_test_sin!(
        tanh_f32,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![0.0, 0.46211717, 0.6640363, 0.7615942],
        f32,
        array_tanh
    );
    impl_test_sin!(
        tanh_f64,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![
            0.0,
            0.46211715726000974,
            0.6640367702678489,
            0.7615941559557649
        ],
        f64,
        array_tanh
    );
    impl_test_sin!(
        abs_f32,
        vec![-1.0, 0.5, -0.8, 1.0],
        vec![1.0, 0.5, 0.8, 1.0],
        f32,
        array_abs
    );
    impl_test_sin!(
        abs_f64,
        vec![-1.0, 0.5, -0.8, 1.0],
        vec![1.0, 0.5, 0.8, 1.0],
        f64,
        array_abs
    );
    impl_test_sin!(
        sqrt_f32,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![0.0, 0.70710677, 0.8944272, 1.0],
        f32,
        array_sqrt
    );
    impl_test_sin!(
        sqrt_f64,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![0.0, 0.7071067811865476, 0.8944271909999159, 1.0],
        f64,
        array_sqrt
    );
    impl_test_sin!(
        exp_f32,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![1.0, 1.6487213, 2.2255409, 2.7182817],
        f32,
        array_exp
    );
    impl_test_sin!(
        exp_f64,
        vec![0.0, 0.5, 0.8, 1.0],
        vec![
            1.0,
            1.6487212707001282,
            2.225540928492468,
            2.718281828459045
        ],
        f64,
        array_exp
    );

    #[test]
    fn set_value_f32() {
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let a_gpu = cuda_malloc(a.len()).unwrap();
        cuda_copy(
            a_gpu,
            a.as_ptr(),
            a.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        set_memory(a_gpu, 1, 1.0);
        let out = get_memory(a_gpu, 1);
        assert_eq!(out, 1.0);
    }

    // fn clip_test() {
    macro_rules! clip_test {
        ($ty:ty) => {
            let a: Vec<$ty> = vec![0.0, 1.0, 2.0, 3.0];
            let mut out = vec![0.0; a.len()];
            let a_gpu = cuda_malloc(a.len()).unwrap();
            cuda_copy(
                a_gpu,
                a.as_ptr(),
                a.len(),
                ZenuCudaMemCopyKind::HostToDevice,
            )
            .unwrap();
            let out_gpu = cuda_malloc(out.len()).unwrap();
            clip(a_gpu, out_gpu, a.len(), 1, 1, 1.0, 2.0);
            cuda_copy(
                out.as_mut_ptr(),
                out_gpu,
                out.len(),
                ZenuCudaMemCopyKind::DeviceToHost,
            )
            .unwrap();
            let ans: Vec<$ty> = vec![1.0, 1.0, 2.0, 2.0];
            assert_eq!(out, ans);

            clip_assign(a_gpu, a.len(), 1, 1.0, 2.0);
            cuda_copy(
                out.as_mut_ptr(),
                a_gpu,
                out.len(),
                ZenuCudaMemCopyKind::DeviceToHost,
            )
            .unwrap();
            assert_eq!(out, ans);
        };
    }
    #[test]
    fn clip_test_f32() {
        clip_test!(f32);
    }
    #[test]
    fn clip_test_f64() {
        clip_test!(f64);
    }
}

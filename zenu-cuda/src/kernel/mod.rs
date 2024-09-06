use std::any::TypeId;

use zenu_cuda_kernel_sys::{
    array_abs_assign_double, array_abs_assign_float, array_abs_double, array_abs_float,
    array_acos_assign_double, array_acos_assign_float, array_acos_double, array_acos_float,
    array_array_add_assign_double, array_array_add_assign_float, array_array_add_double,
    array_array_add_float, array_array_div_assign_double, array_array_div_assign_float,
    array_array_div_double, array_array_div_float, array_array_mul_assign_double,
    array_array_mul_assign_float, array_array_mul_double, array_array_mul_float,
    array_array_sub_assign_double, array_array_sub_assign_float, array_array_sub_double,
    array_array_sub_float, array_asin_assign_double, array_asin_assign_float, array_asin_double,
    array_asin_float, array_atan_assign_double, array_atan_assign_float, array_atan_double,
    array_atan_float, array_clip_assign_double, array_clip_assign_float,
    array_clip_backward_assign_double, array_clip_backward_assign_float,
    array_clip_backward_double, array_clip_backward_float, array_clip_double, array_clip_float,
    array_cos_assign_double, array_cos_assign_float, array_cos_double, array_cos_float,
    array_cosh_assign_double, array_cosh_assign_float, array_cosh_double, array_cosh_float,
    array_exp_assign_double, array_exp_assign_float, array_exp_double, array_exp_float,
    array_log_assign_double, array_log_assign_float, array_log_double, array_log_float,
    array_max_idx_double, array_max_idx_float, array_pow_assign_double, array_pow_assign_float,
    array_pow_double, array_pow_float, array_scalar_add_assign_double,
    array_scalar_add_assign_float, array_scalar_add_double, array_scalar_add_float,
    array_scalar_div_assign_double, array_scalar_div_assign_float, array_scalar_div_double,
    array_scalar_div_float, array_scalar_mul_assign_double, array_scalar_mul_assign_float,
    array_scalar_mul_double, array_scalar_mul_float, array_scalar_pointer_add_assign_double,
    array_scalar_pointer_add_assign_float, array_scalar_pointer_add_double,
    array_scalar_pointer_add_float, array_scalar_pointer_div_assign_double,
    array_scalar_pointer_div_assign_float, array_scalar_pointer_div_double,
    array_scalar_pointer_div_float, array_scalar_pointer_mul_assign_double,
    array_scalar_pointer_mul_assign_float, array_scalar_pointer_mul_double,
    array_scalar_pointer_mul_float, array_scalar_pointer_sub_assign_double,
    array_scalar_pointer_sub_assign_float, array_scalar_pointer_sub_double,
    array_scalar_pointer_sub_float, array_scalar_sub_assign_double, array_scalar_sub_assign_float,
    array_scalar_sub_double, array_scalar_sub_float, array_sin_assign_double,
    array_sin_assign_float, array_sin_double, array_sin_float, array_sinh_assign_double,
    array_sinh_assign_float, array_sinh_double, array_sinh_float, array_sqrt_assign_double,
    array_sqrt_assign_float, array_sqrt_double, array_sqrt_float, array_tan_assign_double,
    array_tan_assign_float, array_tan_double, array_tan_float, array_tanh_assign_double,
    array_tanh_assign_float, array_tanh_double, array_tanh_float, conv_bias_add_double,
    conv_bias_add_float, memory_access_double, memory_access_float, memory_set_double,
    memory_set_float,
};

pub mod activation;

macro_rules! impl_array_scalar {
    ($name:ident, $name_ptr:ident, $double_fn:ident, $float_fn:ident, $double_fn_ptr:ident, $float_fn_ptr:ident) => {
        pub fn $name<T: 'static>(
            out: *mut T,
            a: *const T,
            scalar: T,
            size: usize,
            out_stride: usize,
            stride: usize,
        ) {
            let size = ::libc::c_int::try_from(size).unwrap();
            let stride = ::libc::c_int::try_from(stride).unwrap();
            let out_stride = i32::try_from(out_stride).unwrap();
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a.cast::<f32>().cast_mut();
                let out = out.cast::<f32>();
                let scalar = unsafe { *std::ptr::from_ref(&scalar).cast() };
                unsafe { $float_fn(a, size, stride, scalar, out, out_stride) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a.cast::<f64>().cast_mut();
                let out = out.cast::<f64>();
                let scalar = unsafe { *std::ptr::from_ref(&scalar).cast() };
                unsafe { $double_fn(a, size, stride, scalar, out, out_stride) }
            }
        }

        pub fn $name_ptr<T: 'static>(
            out: *mut T,
            a: *const T,
            scalar: *const T,
            size: usize,
            out_stride: usize,
            stride: usize,
        ) {
            let size = ::libc::c_int::try_from(size).unwrap();
            let stride = ::libc::c_int::try_from(stride).unwrap();
            let out_stride = i32::try_from(out_stride).unwrap();
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a.cast::<f32>().cast_mut();
                let out = out.cast::<f32>();
                let scalar = scalar.cast::<f32>().cast_mut();
                unsafe { $float_fn_ptr(a, size, stride, scalar, out, out_stride) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a.cast::<f64>().cast_mut();
                let out = out.cast::<f64>();
                let scalar = scalar.cast::<f64>().cast_mut();
                unsafe { $double_fn_ptr(a, size, stride, scalar, out, out_stride) }
            }
        }
    };
}
impl_array_scalar!(
    array_scalar_add,
    array_scalar_add_ptr,
    array_scalar_add_double,
    array_scalar_add_float,
    array_scalar_pointer_add_double,
    array_scalar_pointer_add_float
);
impl_array_scalar!(
    array_scalar_sub,
    array_scalar_sub_ptr,
    array_scalar_sub_double,
    array_scalar_sub_float,
    array_scalar_pointer_sub_double,
    array_scalar_pointer_sub_float
);
impl_array_scalar!(
    array_scalar_mul,
    array_scalar_mul_ptr,
    array_scalar_mul_double,
    array_scalar_mul_float,
    array_scalar_pointer_mul_double,
    array_scalar_pointer_mul_float
);
impl_array_scalar!(
    array_scalar_div,
    array_scalar_div_ptr,
    array_scalar_div_double,
    array_scalar_div_float,
    array_scalar_pointer_div_double,
    array_scalar_pointer_div_float
);

macro_rules! impl_array_scalar_assign {
    (
        $name_scalar:ident,
        $name_scalar_ptr: ident,
        $double_fn:ident,
        $float_fn:ident,
        $double_pointer:ident,
        $float_pointer:ident
    ) => {
        pub fn $name_scalar<T: 'static>(a: *mut T, scalar: T, size: usize, stride: usize) {
            let size = ::libc::c_int::try_from(size).unwrap();
            let stride = ::libc::c_int::try_from(stride).unwrap();
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a.cast();
                let scalar = unsafe { *std::ptr::from_ref(&scalar).cast() };
                unsafe { $float_fn(a, size, stride, scalar) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a.cast();
                let scalar = unsafe { *std::ptr::from_ref(&scalar).cast() };
                unsafe { $double_fn(a, size, stride, scalar) }
            }
        }

        pub fn $name_scalar_ptr<T: 'static>(
            a: *mut T,
            scalar: *const T,
            size: usize,
            stride: usize,
        ) {
            let size = ::libc::c_int::try_from(size).unwrap();
            let stride = ::libc::c_int::try_from(stride).unwrap();
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a.cast();
                let scalar = scalar.cast::<f32>().cast_mut();
                unsafe { $float_pointer(a, size, stride, scalar) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a.cast();
                let scalar = scalar.cast::<f64>().cast_mut();
                unsafe { $double_pointer(a, size, stride, scalar) }
            }
        }
    };
}
impl_array_scalar_assign!(
    array_scalar_add_assign,
    array_scalar_add_assign_ptr,
    array_scalar_add_assign_double,
    array_scalar_add_assign_float,
    array_scalar_pointer_add_assign_double,
    array_scalar_pointer_add_assign_float
);
impl_array_scalar_assign!(
    array_scalar_sub_assign,
    array_scalar_sub_assign_ptr,
    array_scalar_sub_assign_double,
    array_scalar_sub_assign_float,
    array_scalar_pointer_sub_assign_double,
    array_scalar_pointer_sub_assign_float
);
impl_array_scalar_assign!(
    array_scalar_mul_assign,
    array_scalar_mul_assign_ptr,
    array_scalar_mul_assign_double,
    array_scalar_mul_assign_float,
    array_scalar_pointer_mul_assign_double,
    array_scalar_pointer_mul_assign_float
);
impl_array_scalar_assign!(
    array_scalar_div_assign,
    array_scalar_div_assign_ptr,
    array_scalar_div_assign_double,
    array_scalar_div_assign_float,
    array_scalar_pointer_div_assign_double,
    array_scalar_pointer_div_assign_float
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
            let size = ::libc::c_int::try_from(size).unwrap();
            let stride_a = ::libc::c_int::try_from(stride_a).unwrap();
            let stride_b = ::libc::c_int::try_from(stride_b).unwrap();
            let stride_c = ::libc::c_int::try_from(stride_c).unwrap();
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a.cast::<f32>().cast_mut();
                let b = b.cast::<f32>().cast_mut();
                let c = c.cast::<f32>();
                unsafe { $float_fn(a, stride_a, b, stride_b, c, stride_c, size) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a.cast::<f64>().cast_mut();
                let b = b.cast::<f64>().cast_mut();
                let c = c.cast::<f64>();
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
            let size = ::libc::c_int::try_from(size).unwrap();
            let stride_a = ::libc::c_int::try_from(stride_a).unwrap();
            let stride_b = ::libc::c_int::try_from(stride_b).unwrap();
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a.cast::<f32>();
                let b = b.cast::<f32>().cast_mut();
                unsafe { $float_fn(a, stride_a, b, stride_b, size) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a.cast::<f64>();
                let b = b.cast::<f64>().cast_mut();
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
        pub fn $name<T: 'static>(
            to: *mut T,
            other: *const T,
            num_elm: usize,
            to_stride: usize,
            other_stride: usize,
        ) {
            let other_stride = ::libc::c_int::try_from(other_stride).unwrap();
            let to_stride = ::libc::c_int::try_from(to_stride).unwrap();
            let num_elm = ::libc::c_int::try_from(num_elm).unwrap();
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let other = other.cast::<f32>().cast_mut();
                let to = to.cast();
                unsafe { $float_fn(other, num_elm, other_stride, to, to_stride) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let other = other.cast::<f64>().cast_mut();
                let to = to.cast();
                unsafe { $double_fn(other, num_elm, other_stride, to, to_stride) }
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
impl_array_scalar_sin!(array_log, array_log_double, array_log_float);

macro_rules! impl_array_scalar_sin_assign {
    ($name:ident, $double_fn:ident, $float_fn:ident) => {
        pub fn $name<T: 'static>(a: *mut T, size: usize, stride: usize) {
            let size = ::libc::c_int::try_from(size).unwrap();
            let stride = ::libc::c_int::try_from(stride).unwrap();
            if TypeId::of::<T>() == TypeId::of::<f32>() {
                let a = a.cast();
                unsafe { $float_fn(a, size, stride) };
            } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                let a = a.cast();
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
impl_array_scalar_sin_assign!(
    array_log_assign,
    array_log_assign_double,
    array_log_assign_float
);

#[allow(clippy::missing_panics_doc)]
pub fn get_memory<T: 'static + Default>(array: *const T, offset: usize) -> T {
    let mut out: T = Default::default();
    let offset = ::libc::c_int::try_from(offset).unwrap();
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let array = array as *mut f32;
        unsafe {
            memory_access_float(array, offset, std::ptr::from_mut(&mut out).cast());
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let array = array as *mut f64;
        unsafe {
            memory_access_double(array, offset, std::ptr::from_mut(&mut out).cast());
        }
    }
    out
}

#[allow(clippy::missing_panics_doc)]
pub fn set_memory<T: 'static + Copy>(array: *mut T, offset: usize, value: T) {
    let offset = ::libc::c_int::try_from(offset).unwrap();
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let array = array.cast::<f32>();
        let value = unsafe {*std::ptr::from_ref(&value).cast::<f32>()};
        unsafe { memory_set_float(array, offset, value) };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let array = array.cast::<f64>();
        let value = unsafe {*std::ptr::from_ref(&value).cast::<f64>()};
        unsafe { memory_set_double(array, offset, value) };
    }
}

#[allow(clippy::missing_panics_doc)]
pub fn clip<T: 'static + Copy>(
    input: *const T,
    output: *mut T,
    size: usize,
    stride_in: usize,
    stride_out: usize,
    min: T,
    max: T,
) {
    let size = ::libc::c_int::try_from(size).unwrap();
    let stride_in = ::libc::c_int::try_from(stride_in).unwrap();
    let stride_out = ::libc::c_int::try_from(stride_out).unwrap();
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let input = input.cast::<f32>().cast_mut();
        let output = output.cast();
        let min = unsafe { *std::ptr::from_ref(&min).cast::<f32>() };
        let max = unsafe { *std::ptr::from_ref(&max).cast::<f32>() };
        unsafe { array_clip_float(input, output, size, stride_in, stride_out, min, max) };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let input = input.cast::<f64>().cast_mut();
        let output = output.cast();
        let min = unsafe { *std::ptr::from_ref(&min).cast::<f64>() };
        let max = unsafe { *std::ptr::from_ref(&max).cast::<f64>() };
        unsafe { array_clip_double(input, output, size, stride_in, stride_out, min, max) };
    }
}

#[allow(clippy::missing_panics_doc)]
pub fn clip_assign<T: 'static + Copy>(input: *mut T, size: usize, stride: usize, min: T, max: T) {
    let size = ::libc::c_int::try_from(size).unwrap();
    let stride = ::libc::c_int::try_from(stride).unwrap();
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let input = unsafe { input.cast::<f32>().as_mut().unwrap() };
        let min = unsafe { *std::ptr::from_ref(&min).cast::<f32>() };
        let max = unsafe { *std::ptr::from_ref(&max).cast::<f32>() };
        unsafe { array_clip_assign_float(input, size, stride, min, max) };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let input = unsafe { input.cast::<f64>().as_mut().unwrap() };
        let min = unsafe { *std::ptr::from_ref(&min).cast::<f64>() };
        let max = unsafe { *std::ptr::from_ref(&max).cast::<f64>() };
        unsafe { array_clip_assign_double(input, size, stride, min, max) };
    }
}

pub fn clip_backward<T: 'static + Copy>(
    input: *mut T,
    mask: *mut T,
    max: T,
    min: T,
    size: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let size = ::libc::c_int::try_from(size).unwrap();
    let stride_in = ::libc::c_int::try_from(stride_in).unwrap();
    let stride_out = ::libc::c_int::try_from(stride_out).unwrap();

    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let input = unsafe { input.cast::<f32>().as_mut().unwrap() };
        let mask = unsafe { mask.cast::<f32>().as_mut().unwrap() };
        let min = unsafe { *std::ptr::from_ref(&min).cast::<f32>() };
        let max = unsafe { *std::ptr::from_ref(&max).cast::<f32>() };
        unsafe { array_clip_backward_float(input, mask, max, min, size, stride_in, stride_out) };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let input = unsafe { input.cast::<f64>().as_mut().unwrap() };
        let mask = unsafe { mask.cast::<f64>().as_mut().unwrap() };
        let min = unsafe { *std::ptr::from_ref(&min).cast::<f64>() };
        let max = unsafe { *std::ptr::from_ref(&max).cast::<f64>() };
        unsafe { array_clip_backward_double(input, mask, max, min, size, stride_in, stride_out) };
    }
}

pub fn clip_backward_assign<T: 'static + Copy>(mask: *mut T, max: T, min: T, size: usize, stride: usize) {
    let size = ::libc::c_int::try_from(size).unwrap();
    let stride = ::libc::c_int::try_from(stride).unwrap();
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let mask = mask.cast();
        let min = unsafe { *std::ptr::from_ref(&min).cast::<f32>() };
        let max = unsafe { *std::ptr::from_ref(&max).cast::<f32>() };
        unsafe { array_clip_backward_assign_float(mask, max, min, size, stride) };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let mask = mask.cast();
        let min = unsafe { *std::ptr::from_ref(&min).cast::<f64>() };
        let max = unsafe { *std::ptr::from_ref(&max).cast::<f64>() };
        unsafe { array_clip_backward_assign_double(mask, max, min, size, stride) };
    }
}

pub fn array_pow<T: 'static + Copy>(
    input: *const T,
    size: usize,
    stride_a: usize,
    scalar: T,
    out: *mut T,
    stride_out: usize,
) {
    let size = ::libc::c_int::try_from(size).unwrap();
    let stride_a = ::libc::c_int::try_from(stride_a).unwrap();
    let stride_out = ::libc::c_int::try_from(stride_out).unwrap();

    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let input = input.cast::<f32>().cast_mut();
        let out = out.cast::<f32>();
        let scalar = unsafe { *std::ptr::from_ref(&scalar).cast::<f32>() };
        unsafe { array_pow_float(input, size, stride_a, scalar, out, stride_out) };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let input = input.cast::<f64>().cast_mut();
        let out = out.cast::<f64>();
        let scalar = unsafe { *std::ptr::from_ref(&scalar).cast::<f64>() };
        unsafe { array_pow_double(input, size, stride_a, scalar, out, stride_out) };
    }
}

pub fn array_pow_assign<T: 'static + Copy>(input: *mut T, size: usize, stride: usize, scalar: T) {
    let size = ::libc::c_int::try_from(size).unwrap();
    let stride = ::libc::c_int::try_from(stride).unwrap();
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let input = unsafe { input.cast::<f32>().as_mut().unwrap() };
        let scalar = unsafe { *std::ptr::from_ref(&scalar).cast::<f32>() };
        unsafe { array_pow_assign_float(input, size, stride, scalar) };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let input = unsafe { input.cast::<f64>().as_mut().unwrap() };
        let scalar = unsafe { *std::ptr::from_ref(&scalar).cast::<f64>() };
        unsafe { array_pow_assign_double(input, size, stride, scalar) };
    }
}

pub fn array_max_idx<T: 'static>(input: *const T, size: usize, stride: usize) -> usize {
    let size = ::libc::c_int::try_from(size).unwrap();
    let stride = ::libc::c_int::try_from(stride).unwrap();
    let mut ans: i32 = 0;
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let input = input.cast::<f32>().cast_mut();
        unsafe {
            array_max_idx_float(
                input,
                size,
                stride,
                std::ptr::from_mut(&mut ans).cast(),
            );
        };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let input = input.cast::<f64>().cast_mut();
        unsafe {
            array_max_idx_double(
                input,
                size,
                stride,
                std::ptr::from_mut(&mut ans).cast(),
            );
        };
    } else {
        panic!("Not supported type");
    }
    usize::try_from(ans).unwrap()
}

pub fn conv_bias_add<T: 'static>(
    input: *const T,
    bias: *const T,
    total_elements: usize,
    channel_stride: usize,
    bias_size: usize,
    output: *mut T,
) {
    let total_elements = ::libc::c_int::try_from(total_elements).unwrap();
    let channel_stride = ::libc::c_int::try_from(channel_stride).unwrap();
    let bias_size = ::libc::c_int::try_from(bias_size).unwrap();
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // let input = input as *mut f32;
        // let bias = bias as *mut f32;
        // let output = output as *mut f32;
        let input = input.cast::<f32>().cast_mut();
        let bias = bias.cast::<f32>().cast_mut();
        let output = unsafe { output.cast::<f32>().as_mut().unwrap() };
        unsafe {
            conv_bias_add_float(
                input,
                output,
                channel_stride,
                bias,
                bias_size,
                total_elements,
            );
        };
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let input = input.cast::<f64>().cast_mut();
        let bias = bias.cast::<f64>().cast_mut();
        let output = unsafe { output.cast::<f64>().as_mut().unwrap() };
        unsafe {
            conv_bias_add_double(
                input,
                output,
                channel_stride,
                bias,
                bias_size,
                total_elements,
            );
        };
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
                let zero: $ty = 0.;
                let mut out = vec![zero; a.len()];
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
                let zero: $ty = 0.;
                let mut out = vec![zero; a.len()];
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

#[allow(clippy::unreadable_literal, clippy::approx_constant, clippy::excessive_precision)]
#[cfg(test)]
mod array_scalar {
    
    use crate::runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind};

    use super::*;

    macro_rules! impl_array_scalr_test {
        ($test_name:ident, $input:expr, $scalar:expr, $ans:expr, $ty:ty, $kernel_func:ident) => {
            #[test]
            fn $test_name() {
                let a: Vec<$ty> = $input;
                let zero: $ty = 0.;
                let mut out = vec![zero; a.len()];
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
                $func(out_gpu, a_gpu, a.len(), 1, 1);
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
    impl_test_sin!(
        log_f32,
        vec![1.0, 2.0, 3.0],
        vec![0.0, 0.6931472, 1.0986123],
        f32,
        array_log
    );
    impl_test_sin!(
        log_f64,
        vec![1.0, 2.0, 3.0],
        vec![0.0, 0.6931471805599453, 1.0986122886681098],
        f64,
        array_log
    );

    #[allow(clippy::float_cmp)]
    #[test]
    fn set_value_f32() {
        let a = [0.0, 0.0, 0.0, 0.0];
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

    macro_rules! clip_backward {
        ($ty:ty) => {
            let a: Vec<$ty> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
            let mut out = vec![0.0; a.len()];
            let a_gpu = cuda_malloc(a.len()).unwrap();
            cuda_copy(
                a_gpu,
                a.as_ptr(),
                a.len(),
                ZenuCudaMemCopyKind::HostToDevice,
            )
            .unwrap();
            clip_backward(a_gpu, a_gpu, 2.0, 1.0, a.len(), 1, 1);
            cuda_copy(
                out.as_mut_ptr(),
                a_gpu,
                out.len(),
                ZenuCudaMemCopyKind::DeviceToHost,
            )
            .unwrap();
            let ans: Vec<$ty> = vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
            assert_eq!(out, ans);
        };
    }
    #[test]
    fn clip_backward_f32() {
        clip_backward!(f32);
    }
    #[test]
    fn clip_backward_f64() {
        clip_backward!(f64);
    }

    macro_rules! clip_backward_assign {
        ($ty:ty) => {
            let a: Vec<$ty> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
            let mut out = vec![0.0; a.len()];
            let a_gpu = cuda_malloc(a.len()).unwrap();
            cuda_copy(
                a_gpu,
                a.as_ptr(),
                a.len(),
                ZenuCudaMemCopyKind::HostToDevice,
            )
            .unwrap();
            clip_backward_assign(a_gpu, 2.0, 1.0, a.len(), 1);
            cuda_copy(
                out.as_mut_ptr(),
                a_gpu,
                out.len(),
                ZenuCudaMemCopyKind::DeviceToHost,
            )
            .unwrap();
            let ans: Vec<$ty> = vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
            assert_eq!(out, ans);
        };
    }
    #[test]
    fn clip_backward_assign_f32() {
        clip_backward_assign!(f32);
    }
    #[test]
    fn clip_backward_assign_f64() {
        clip_backward_assign!(f64);
    }

    macro_rules! pow_test {
        ($ty:ty) => {
            let a: Vec<$ty> = vec![0.0, 1.0, 2.0, 3.0];
            let mut out = vec![0.0; a.len()];
            let a_gpu = cuda_malloc(a.len()).unwrap();
            let out_gpu = cuda_malloc(out.len()).unwrap();
            cuda_copy(
                a_gpu,
                a.as_ptr(),
                a.len(),
                ZenuCudaMemCopyKind::HostToDevice,
            )
            .unwrap();
            array_pow(a_gpu, a.len(), 1, 2.0, out_gpu, 1);
            cuda_copy(
                out.as_mut_ptr(),
                out_gpu,
                out.len(),
                ZenuCudaMemCopyKind::DeviceToHost,
            )
            .unwrap();
            let ans: Vec<$ty> = vec![0.0, 1.0, 4.0, 9.0];
            assert_eq!(out, ans);

            array_pow_assign(a_gpu, a.len(), 1, 2.0);
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
    fn pow_test_f32() {
        pow_test!(f32);
    }
    #[test]
    fn pow_test_f64() {
        pow_test!(f64);
    }

    macro_rules! pow_assign {
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
            array_pow_assign(a_gpu, a.len(), 1, 2.0);
            cuda_copy(
                out.as_mut_ptr(),
                a_gpu,
                out.len(),
                ZenuCudaMemCopyKind::DeviceToHost,
            )
            .unwrap();
            let ans: Vec<$ty> = vec![0.0, 1.0, 4.0, 9.0];
            assert_eq!(out, ans);
        };
    }
    #[test]
    fn pow_assign_f32() {
        pow_assign!(f32);
    }
    #[test]
    fn pow_assign_f64() {
        pow_assign!(f64);
    }
}

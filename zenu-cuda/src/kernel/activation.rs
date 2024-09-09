use std::any::TypeId;

use zenu_cuda_kernel_sys::{
    relu_backward_mask_double, relu_backward_mask_float, relu_double, relu_float,
};

#[expect(clippy::missing_panics_doc)]
pub fn relu<T: 'static + Copy>(
    input: *const T,
    output: *mut T,
    alpha: T,
    size: usize,
    input_stride: usize,
    output_stride: usize,
) {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let alpha = unsafe { *std::ptr::from_ref(&alpha).cast() };
        unsafe {
            relu_float(
                input.cast::<f32>().cast_mut(),
                output.cast(),
                alpha,
                ::libc::c_int::try_from(size).unwrap(),
                ::libc::c_int::try_from(input_stride).unwrap(),
                ::libc::c_int::try_from(output_stride).unwrap(),
            );
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let alpha = unsafe { *std::ptr::from_ref(&alpha).cast() };
        unsafe {
            relu_double(
                input.cast::<f64>().cast_mut(),
                output.cast(),
                alpha,
                ::libc::c_int::try_from(size).unwrap(),
                ::libc::c_int::try_from(input_stride).unwrap(),
                ::libc::c_int::try_from(output_stride).unwrap(),
            );
        }
    } else {
        panic!("Unsupported data type");
    }
}

#[expect(clippy::missing_panics_doc)]
pub fn relu_backward_mask<T: 'static + Copy>(
    input: *const T,
    mask: *mut T,
    alpha: T,
    size: usize,
    input_stride: usize,
    mask_stride: usize,
) {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let alpha = unsafe { *std::ptr::from_ref(&alpha).cast() };
        unsafe {
            relu_backward_mask_float(
                input.cast::<f32>().cast_mut(),
                mask.cast(),
                alpha,
                ::libc::c_int::try_from(size).unwrap(),
                ::libc::c_int::try_from(input_stride).unwrap(),
                ::libc::c_int::try_from(mask_stride).unwrap(),
            );
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let alpha = unsafe { *std::ptr::from_ref(&alpha).cast() };
        unsafe {
            relu_backward_mask_double(
                input.cast::<f64>().cast_mut(),
                mask.cast(),
                alpha,
                ::libc::c_int::try_from(size).unwrap(),
                ::libc::c_int::try_from(input_stride).unwrap(),
                ::libc::c_int::try_from(mask_stride).unwrap(),
            );
        }
    } else {
        panic!("Unsupported data type");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind};

    #[test]
    fn test_relu() {
        let input: Vec<f32> = vec![1.0, -1.0, 0.0, 2.0, -2.0];
        let mut output: Vec<f32> = input.clone();
        let alpha = 0.2;
        let size = input.len();

        let input_gpu = cuda_malloc::<f32>(size).unwrap();
        let output_gpu = cuda_malloc::<f32>(size).unwrap();

        cuda_copy(
            input_gpu,
            input.as_ptr(),
            size,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        relu(input_gpu, output_gpu, alpha, size, 1, 1);

        cuda_copy(
            output.as_mut_ptr(),
            output_gpu,
            size,
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();
        let ans: Vec<f32> = vec![1.0, -0.2, 0.0, 2.0, -0.4];
        assert_eq!(output, ans);
    }

    #[test]
    fn test_relu_backward_mask() {
        let input: Vec<f32> = vec![1.0, -1.0, 0.0, 2.0, -2.0];
        let mut mask: Vec<f32> = input.clone();
        let size = input.len();

        let input_gpu = cuda_malloc::<f32>(size).unwrap();
        let mask_gpu = cuda_malloc::<f32>(size).unwrap();

        cuda_copy(
            input_gpu,
            input.as_ptr(),
            size,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        relu_backward_mask(input_gpu, mask_gpu, 0.1, size, 1, 1);

        cuda_copy(
            mask.as_mut_ptr(),
            mask_gpu,
            size,
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();
        let ans: Vec<f32> = vec![1.0, -0.1, -0.1, 1.0, -0.1];
        assert_eq!(mask, ans);
    }
}

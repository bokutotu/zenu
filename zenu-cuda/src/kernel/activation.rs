use std::any::TypeId;

use zenu_cuda_kernel_sys::*;

pub fn relu<T: 'static>(
    input: *const T,
    output: *mut T,
    alpha: T,
    size: usize,
    input_stride: usize,
    output_stride: usize,
) {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let alpha: f32 = unsafe { *(&alpha as *const T as *const f32) };
        unsafe {
            relu_float(
                input as *mut f32,
                output as *mut f32,
                alpha as f32,
                size as ::libc::c_int,
                input_stride as ::libc::c_int,
                output_stride as ::libc::c_int,
            )
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let alpha: f64 = unsafe { *(&alpha as *const T as *const f64) };
        unsafe {
            relu_double(
                input as *mut f64,
                output as *mut f64,
                alpha as f64,
                size as ::libc::c_int,
                input_stride as ::libc::c_int,
                output_stride as ::libc::c_int,
            )
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
}

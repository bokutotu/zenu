use std::{any::TypeId, ptr::NonNull};

use zenu_cublas_sys::{cublasDcopy_v2, cublasScopy_v2};

use crate::ZENU_CUDA_STATE;

use self::cublas_error::ZenuCublasError;

pub mod cublas_error;

pub fn cublas_copy<T: 'static>(
    n: usize,
    x: NonNull<T>,
    incx: usize,
    y: NonNull<T>,
    incy: usize,
) -> Result<(), ZenuCublasError> {
    let context = ZENU_CUDA_STATE.lock().unwrap();
    let cublas_context = context.get_cublas();
    let err = if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            cublasScopy_v2(
                cublas_context.as_ptr(),
                n as i32,
                x.as_ptr() as *const f32,
                incx as i32,
                y.as_ptr() as *mut f32,
                incy as i32,
            )
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            cublasDcopy_v2(
                cublas_context.as_ptr(),
                n as i32,
                x.as_ptr() as *const f64,
                incx as i32,
                y.as_ptr() as *mut f64,
                incy as i32,
            )
        }
    } else {
        panic!("Unsupported type");
    };
    match ZenuCublasError::from(err as u32) {
        ZenuCublasError::CublasStatusSuccess => Ok(()),
        err => Err(err),
    }
}

#[cfg(test)]
mod cublas {
    use std::ptr::NonNull;

    use crate::runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind};

    use super::cublas_copy;

    #[test]
    fn cublas_copy_small() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

        let x_gpu = cuda_malloc(x.len()).unwrap();
        let y_gpu = cuda_malloc(y.len()).unwrap();

        cuda_copy(
            x_gpu,
            unsafe { NonNull::new_unchecked(x.as_ptr() as *mut f32) },
            x.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        cublas_copy(x.len(), x_gpu, 1, y_gpu, 1).unwrap();

        cuda_copy(
            unsafe { NonNull::new_unchecked(y.as_ptr() as *mut f32) },
            y_gpu,
            y.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();

        assert_eq!(x, y);
    }
}

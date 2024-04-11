use std::{any::TypeId, ptr::NonNull};

use zenu_cublas_sys::{
    cublasDcopy_v2, cublasDgemm_v2_64, cublasOperation_t, cublasScopy_v2, cublasSgemm_v2,
    cublasSgemm_v2_64,
};

use crate::ZENU_CUDA_STATE;

use self::cublas_error::ZenuCublasError;

pub mod cublas_error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZenuCublasOperation {
    N,
    T,
    C,
    ConjT,
}

impl From<ZenuCublasOperation> for cublasOperation_t {
    fn from(value: ZenuCublasOperation) -> Self {
        match value {
            ZenuCublasOperation::N => cublasOperation_t::CUBLAS_OP_N,
            ZenuCublasOperation::T => cublasOperation_t::CUBLAS_OP_T,
            ZenuCublasOperation::C => cublasOperation_t::CUBLAS_OP_C,
            ZenuCublasOperation::ConjT => cublasOperation_t::CUBLAS_OP_CONJG,
        }
    }
}

pub fn cublas_copy<T: 'static>(
    n: usize,
    x: *const T,
    incx: usize,
    y: *mut T,
    incy: usize,
) -> Result<(), ZenuCublasError> {
    let context = ZENU_CUDA_STATE.lock().unwrap();
    let cublas_context = context.get_cublas();
    let err = if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            cublasScopy_v2(
                cublas_context.as_ptr(),
                n as i32,
                x as *const f32,
                incx as i32,
                y as *mut f32,
                incy as i32,
            )
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            cublasDcopy_v2(
                cublas_context.as_ptr(),
                n as i32,
                x as *const f64,
                incx as i32,
                y as *mut f64,
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

pub fn cublas_gemm<T: 'static>(
    transa: ZenuCublasOperation,
    transb: ZenuCublasOperation,
    m: i32,
    n: i32,
    k: i32,
    alpha: T,
    a: *const T,
    lda: i32,
    b: *const T,
    ldb: i32,
    beta: T,
    c: *mut T,
    ldc: i32,
) -> Result<(), ZenuCublasError> {
    let transa = cublasOperation_t::from(transa);
    let transb = cublasOperation_t::from(transb);
    let context = ZENU_CUDA_STATE.lock().unwrap();
    let cublas_context = context.get_cublas();
    let err = if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            cublasSgemm_v2_64(
                cublas_context.as_ptr(),
                transa,
                transb,
                m as i64,
                n as i64,
                k as i64,
                &alpha as *const T as *const f32,
                a as *const f32,
                lda as i64,
                b as *const f32,
                ldb as i64,
                &beta as *const T as *const f32,
                c as *mut f32,
                ldc as i64,
            )
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            cublasDgemm_v2_64(
                cublas_context.as_ptr(),
                transa,
                transb,
                m as i64,
                n as i64,
                k as i64,
                &alpha as *const T as *const f64,
                a as *const f64,
                lda as i64,
                b as *const f64,
                ldb as i64,
                &beta as *const T as *const f64,
                c as *mut f64,
                ldc as i64,
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
    use crate::runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind};

    use super::cublas_copy;

    #[test]
    fn cublas_copy_small() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut y: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

        let x_gpu = cuda_malloc(x.len()).unwrap();
        let y_gpu = cuda_malloc(y.len()).unwrap();

        cuda_copy(
            x_gpu,
            x.as_ptr(),
            x.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        cublas_copy(x.len(), x_gpu, 1, y_gpu, 1).unwrap();

        cuda_copy(
            y.as_mut_ptr(),
            y_gpu,
            y.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();

        assert_eq!(x, y);
    }
}

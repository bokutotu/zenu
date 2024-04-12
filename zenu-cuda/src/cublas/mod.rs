use std::{any::TypeId, ptr::NonNull};

use zenu_cublas_sys::{
    cublasDasum_v2, cublasDasum_v2_64, cublasDcopy_v2, cublasDgemm_v2_64, cublasOperation_t,
    cublasSasum_v2_64, cublasScopy_v2, cublasSgemm_v2, cublasSgemm_v2_64,
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

pub fn cublas_asum<T: Default + 'static>(
    n: usize,
    x: *const T,
    incx: usize,
) -> Result<T, ZenuCublasError> {
    let context = ZENU_CUDA_STATE.lock().unwrap();
    let cublas_context = context.get_cublas();
    let mut result: T = Default::default();
    let err = if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            cublasSasum_v2_64(
                cublas_context.as_ptr(),
                n as i64,
                x as *const f32,
                incx as i64,
                &mut result as *mut T as *mut f32,
            )
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            cublasDasum_v2_64(
                cublas_context.as_ptr(),
                n as i64,
                x as *const f64,
                incx as i64,
                &mut result as *mut T as *mut f64,
            )
        }
    } else {
        panic!("Unsupported type");
    };

    match ZenuCublasError::from(err as u32) {
        ZenuCublasError::CublasStatusSuccess => Ok(result),
        err => Err(err),
    }
}

#[cfg(test)]
mod cublas {
    use crate::{
        cublas::{cublas_gemm, ZenuCublasOperation},
        runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind},
    };

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

    #[test]
    fn gemm_f32() {
        let m = 2;
        let n = 2;
        let k = 2;

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; m * n];

        let a_gpu = cuda_malloc(a.len()).unwrap();
        let b_gpu = cuda_malloc(b.len()).unwrap();
        let c_gpu = cuda_malloc(c.len()).unwrap();

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

        cublas_gemm(
            ZenuCublasOperation::N,
            ZenuCublasOperation::N,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a_gpu,
            m as i32,
            b_gpu,
            k as i32,
            0.0,
            c_gpu,
            m as i32,
        )
        .unwrap();

        cuda_copy(
            c.as_mut_ptr(),
            c_gpu,
            c.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();

        assert_eq!(c, vec![7., 10., 15., 22.]);
    }

    #[test]
    fn gemm_f64_both_column_major() {
        // shape (3, 4)
        // let a: Vec<f64> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let a: Vec<f64> = vec![1., 5., 9., 2., 6., 10., 3., 7., 11., 4., 8., 12.];
        // shape (4, 2)
        // let b = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let b = vec![1., 3., 5., 7., 2., 4., 6., 8.];
        // shape (3, 2)
        let mut c = vec![0.0; 6];

        let a_gpu = cuda_malloc(a.len()).unwrap();
        let b_gpu = cuda_malloc(b.len()).unwrap();
        let c_gpu = cuda_malloc(c.len()).unwrap();

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

        cuda_copy(
            c_gpu,
            c.as_ptr(),
            c.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        cublas_gemm(
            ZenuCublasOperation::N,
            ZenuCublasOperation::N,
            3,
            2,
            4,
            1.0,
            a_gpu,
            3,
            b_gpu,
            4,
            0.0,
            c_gpu,
            3,
        )
        .unwrap();

        cuda_copy(
            c.as_mut_ptr(),
            c_gpu,
            c.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();

        // assert_eq!(c, vec![50., 60., 114., 140., 178., 220.]);
        assert_eq!(c, vec![50., 114., 178., 60., 140., 220.]);
    }

    #[test]
    fn gemm_f64_both_row_major_outpu_column_major() {
        // shape (3, 4)
        let a: Vec<f64> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        // shape (4, 2)
        let b = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        // shape (3, 2)
        let mut c = vec![0.0; 6];

        let a_gpu = cuda_malloc(a.len()).unwrap();
        let b_gpu = cuda_malloc(b.len()).unwrap();
        let c_gpu = cuda_malloc(c.len()).unwrap();

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

        cuda_copy(
            c_gpu,
            c.as_ptr(),
            c.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        cublas_gemm(
            ZenuCublasOperation::T,
            ZenuCublasOperation::T,
            3,
            2,
            4,
            1.0,
            a_gpu,
            4,
            b_gpu,
            2,
            0.0,
            c_gpu,
            3,
        )
        .unwrap();

        cuda_copy(
            c.as_mut_ptr(),
            c_gpu,
            c.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();

        assert_eq!(c, vec![50., 114., 178., 60., 140., 220.]);
    }

    #[test]
    fn axum_f32() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let x_gpu = cuda_malloc(x.len()).unwrap();
        cuda_copy(
            x_gpu,
            x.as_ptr(),
            x.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        let result = super::cublas_asum(x.len(), x_gpu, 1).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn axum_f64() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x_gpu = cuda_malloc(x.len()).unwrap();

        cuda_copy(
            x_gpu,
            x.as_ptr(),
            x.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        let result = super::cublas_asum(x.len(), x_gpu, 1).unwrap();
        assert_eq!(result, 10.0);
    }
}

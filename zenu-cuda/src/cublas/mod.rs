use std::any::TypeId;

use zenu_cublas_sys::{
    cublasDasum_v2_64, cublasDcopy_v2, cublasDdot_v2_64, cublasDgemm_v2_64, cublasIdamax_v2_64,
    cublasIsamax_v2_64, cublasOperation_t, cublasSasum_v2_64, cublasScopy_v2, cublasSdot_v2_64,
    cublasSgemm_v2_64,
};

use crate::ZENU_CUDA_STATE;

use self::cublas_error::ZenuCublasError;

#[allow(clippy::module_name_repetitions)]
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

#[allow(clippy::similar_names)]
pub fn cublas_copy<T: 'static>(
    n: usize,
    x: *const T,
    incx: usize,
    y: *mut T,
    incy: usize,
) -> Result<(), ZenuCublasError> {
    let context = ZENU_CUDA_STATE.lock().unwrap();
    let cublas_context = context.get_cublas();
    let n = i32::try_from(n).unwrap();
    let incx = i32::try_from(incx).unwrap();
    let incy = i32::try_from(incy).unwrap();
    let err = if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            cublasScopy_v2(
                cublas_context.as_ptr(),
                n,
                x.cast::<f32>(),
                incx,
                y.cast::<f32>(),
                incy,
            )
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            cublasDcopy_v2(
                cublas_context.as_ptr(),
                n,
                x.cast::<f64>(),
                incx,
                y.cast::<f64>(),
                incy,
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

#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::many_single_char_names,
)]
pub fn cublas_gemm<T: 'static + Copy>(
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
    let context = ZENU_CUDA_STATE
        .lock()
        .expect("Failed to lock ZENU_CUDA_STATE this is maybe a bug");
    let cublas_context = context.get_cublas();

    let m = i64::from(m);
    let n = i64::from(n);
    let k = i64::from(k);

    let lda = i64::from(lda);
    let ldb = i64::from(ldb);
    let ldc = i64::from(ldc);

    let err = if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            cublasSgemm_v2_64(
                cublas_context.as_ptr(),
                transa,
                transb,
                m,
                n,
                k,
                std::ptr::from_ref(&alpha).cast::<f32>(),
                a.cast::<f32>(),
                lda,
                b.cast::<f32>(),
                ldb,
                std::ptr::from_ref(&beta).cast::<f32>(),
                c.cast::<f32>(),
                ldc,
            )
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            cublasDgemm_v2_64(
                cublas_context.as_ptr(),
                transa,
                transb,
                m,
                n,
                k,
                std::ptr::from_ref(&alpha).cast::<f64>(),
                a.cast::<f64>(),
                lda,
                b.cast::<f64>(),
                ldb,
                std::ptr::from_ref(&beta).cast::<f64>(),
                c.cast::<f64>(),
                ldc,
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

    let n = i64::try_from(n).unwrap();
    let incx = i64::try_from(incx).unwrap();

    let err = if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            cublasSasum_v2_64(
                cublas_context.as_ptr(),
                n,
                x.cast::<f32>(),
                incx,
                std::ptr::from_mut(&mut result).cast::<f32>(),
            )
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            cublasDasum_v2_64(
                cublas_context.as_ptr(),
                n,
                x.cast::<f64>(),
                incx,
                std::ptr::from_mut(&mut result).cast::<f64>(),
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

pub fn cublas_amax<T: Default + 'static>(
    n: usize,
    x: *const T,
    incx: usize,
) -> Result<i64, ZenuCublasError> {
    let context = ZENU_CUDA_STATE.lock().unwrap();
    let cublas_context = context.get_cublas();
    let mut result: i64 = 0;

    let n = i64::try_from(n).unwrap();
    let incx = i64::try_from(incx).unwrap();

    let err = if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            cublasIsamax_v2_64(
                cublas_context.as_ptr(),
                n,
                x.cast::<f32>(),
                incx,
                std::ptr::from_mut(&mut result),
            ) as i32
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            cublasIdamax_v2_64(
                cublas_context.as_ptr(),
                n,
                x.cast::<f64>(),
                incx,
                std::ptr::from_mut(&mut result),
            ) as i32
        }
    } else {
        panic!("Unsupported type");
    };

    match ZenuCublasError::from(u32::try_from(err).unwrap()) {
        ZenuCublasError::CublasStatusSuccess => Ok(result - 1),
        err => Err(err),
    }
}

#[allow(clippy::similar_names)]
pub fn cublas_dot<T: 'static + Default>(
    n: usize,
    x: *const T,
    incx: usize,
    y: *const T,
    incy: usize,
) -> Result<T, ZenuCublasError> {
    let context = ZENU_CUDA_STATE.lock().unwrap();
    let cublas_context = context.get_cublas();
    let mut result: T = Default::default();

    let n = i64::try_from(n).unwrap();
    let incx = i64::try_from(incx).unwrap();
    let incy = i64::try_from(incy).unwrap();

    let err = if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            cublasSdot_v2_64(
                cublas_context.as_ptr(),
                n,
                x.cast::<f32>(),
                incx,
                y.cast::<f32>(),
                incy,
                std::ptr::from_mut(&mut result).cast::<f32>(),
            )
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            cublasDdot_v2_64(
                cublas_context.as_ptr(),
                n,
                x.cast::<f64>(),
                incx,
                y.cast::<f64>(),
                incy,
                std::ptr::from_mut(&mut result).cast::<f64>(),
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

pub fn cublas_scal<T: 'static + Copy>(
    n: usize,
    alpha: T,
    x: *mut T,
    incx: usize,
) -> Result<(), ZenuCublasError> {
    let context = ZENU_CUDA_STATE.lock().unwrap();
    let cublas_context = context.get_cublas();

    let n = i32::try_from(n).unwrap();
    let incx = i32::try_from(incx).unwrap();

    let err = if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe {
            zenu_cublas_sys::cublasSscal_v2(
                cublas_context.as_ptr(),
                n,
                std::ptr::from_ref(&alpha).cast::<f32>(),
                x.cast::<f32>(),
                incx,
            )
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe {
            zenu_cublas_sys::cublasDscal_v2(
                cublas_context.as_ptr(),
                n,
                std::ptr::from_ref(&alpha).cast::<f64>(),
                x.cast::<f64>(),
                incx,
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
mod cublas_tests {
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
    #[allow(clippy::many_single_char_names)]
    fn gemm_f32() {
        let m: i32 = 2;
        let n: i32 = 2;
        let k: i32 = 2;

        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; (m * n).try_into().unwrap()];

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
            m,
            n,
            k,
            1.0,
            a_gpu,
            m,
            b_gpu,
            k,
            0.0,
            c_gpu,
            m,
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
        let b = [1., 3., 5., 7., 2., 4., 6., 8.];
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
        let b = [1., 2., 3., 4., 5., 6., 7., 8.];
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

    #[allow(clippy::float_cmp)]
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

    #[allow(clippy::float_cmp)]
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

    #[allow(clippy::float_cmp)]
    #[test]
    fn amax_f32() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let x_gpu = cuda_malloc(x.len()).unwrap();

        cuda_copy(
            x_gpu,
            x.as_ptr(),
            x.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        let result = super::cublas_amax(x.len(), x_gpu, 1).unwrap();
        assert_eq!(result, 3);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn amax_f64() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x_gpu = cuda_malloc(x.len()).unwrap();

        cuda_copy(
            x_gpu,
            x.as_ptr(),
            x.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        let result = super::cublas_amax(x.len(), x_gpu, 1).unwrap();
        assert_eq!(result, 3);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn dot_f32() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let x_gpu = cuda_malloc(x.len()).unwrap();
        let y_gpu = cuda_malloc(y.len()).unwrap();

        cuda_copy(
            x_gpu,
            x.as_ptr(),
            x.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        cuda_copy(
            y_gpu,
            y.as_ptr(),
            y.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        let result = super::cublas_dot(x.len(), x_gpu, 1, y_gpu, 1).unwrap();
        assert_eq!(result, 30.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn dot_f64() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x_gpu = cuda_malloc(x.len()).unwrap();
        let y_gpu = cuda_malloc(y.len()).unwrap();

        cuda_copy(
            x_gpu,
            x.as_ptr(),
            x.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        cuda_copy(
            y_gpu,
            y.as_ptr(),
            y.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        let result = super::cublas_dot(x.len(), x_gpu, 1, y_gpu, 1).unwrap();
        assert_eq!(result, 30.0);
    }
}

pub mod cublas;
pub mod cudnn;
pub mod kernel;
pub mod runtime;

use std::{ptr::NonNull, sync::Mutex};

use cublas::cublas_error::ZenuCublasError;
use cudnn::error::ZenuCudnnError;
use once_cell::sync::Lazy;
use zenu_cublas_sys::{cublasContext, cublasCreate_v2, cublasDestroy_v2};
use zenu_cudnn_sys::{cudnnContext, cudnnCreate, cudnnDestroy};

static ZENU_CUDA_STATE: Lazy<Mutex<ZenuCudaState>> = Lazy::new(|| Mutex::new(ZenuCudaState::new()));

pub struct ZenuCudaState {
    cublas: NonNull<cublasContext>,
    cudnn: NonNull<cudnnContext>,
}

unsafe impl Send for ZenuCudaState {}
unsafe impl Sync for ZenuCudaState {}

fn create_cublas_context() -> Result<NonNull<cublasContext>, ()> {
    let mut null_ptr = std::ptr::null_mut();
    let null_ptr_ptr = &mut null_ptr as *mut *mut cublasContext;
    let cublas_handel = NonNull::new(null_ptr_ptr).unwrap();
    let err: u32 = unsafe { cublasCreate_v2(cublas_handel.as_ptr()) as u32 };
    let err = ZenuCublasError::from(err);
    match err {
        ZenuCublasError::CublasStatusSuccess => {
            let cublas = unsafe { *cublas_handel.as_ptr() };
            let cublas = NonNull::new(cublas).unwrap();
            Ok(cublas)
        }
        _ => Err(()),
    }
}

fn create_cudnn_context() -> Result<NonNull<cudnnContext>, ()> {
    let mut null_ptr = std::ptr::null_mut();
    let null_ptr_ptr = &mut null_ptr as *mut *mut cudnnContext;
    let cudnn_handel = NonNull::new(null_ptr_ptr).unwrap();
    let err: u32 = unsafe { cudnnCreate(cudnn_handel.as_ptr()) as u32 };
    if err != 0 {
        Err(())
    } else {
        let cudnn = unsafe { *cudnn_handel.as_ptr() };
        let cudnn = NonNull::new(cudnn).unwrap();
        Ok(cudnn)
    }
}

impl ZenuCudaState {
    fn new() -> Self {
        let cublas = create_cublas_context().unwrap();
        let cudnn = create_cudnn_context().unwrap();
        Self { cublas, cudnn }
    }

    pub fn get_cublas(&self) -> NonNull<cublasContext> {
        self.cublas
    }

    pub fn get_cudnn(&self) -> NonNull<cudnnContext> {
        self.cudnn
    }
}

impl Drop for ZenuCudaState {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.cublas.as_ptr());
        }

        unsafe {
            cudnnDestroy(self.cudnn.as_ptr());
        }
    }
}

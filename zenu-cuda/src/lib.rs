pub mod cublas;
pub mod kernel;
pub mod runtime;

use std::{ptr::NonNull, sync::Mutex};

use cublas::cublas_error::ZenuCublasError;
use once_cell::sync::Lazy;
use zenu_cublas_sys::{cublasContext, cublasCreate_v2, cublasDestroy_v2};

static ZENU_CUDA_STATE: Lazy<Mutex<ZenuCudaState>> = Lazy::new(|| Mutex::new(ZenuCudaState::new()));

pub struct ZenuCudaState {
    cublas: NonNull<cublasContext>,
}

unsafe impl Send for ZenuCudaState {}
unsafe impl Sync for ZenuCudaState {}

impl ZenuCudaState {
    fn new() -> Self {
        let mut null_ptr = std::ptr::null_mut();
        let null_ptr_ptr = &mut null_ptr as *mut *mut cublasContext;
        let cublas_handel = NonNull::new(null_ptr_ptr).unwrap();
        let err: u32 =
            unsafe { cublasCreate_v2(cublas_handel.as_ptr() as *mut *mut cublasContext) as u32 };
        let err = ZenuCublasError::from(err);
        match err {
            ZenuCublasError::CublasStatusSuccess => {
                let cublas = unsafe { *cublas_handel.as_ptr() };
                let cublas = NonNull::new(cublas).unwrap();
                ZenuCudaState { cublas }
            }
            _ => panic!("Failed to create cublas handel"),
        }
    }

    pub fn get_cublas(&self) -> NonNull<cublasContext> {
        self.cublas
    }
}

impl Drop for ZenuCudaState {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.cublas.as_ptr());
        }
    }
}

#![expect(clippy::missing_panics_doc, clippy::missing_errors_doc)]
#![expect(clippy::module_name_repetitions)]

pub mod cublas;
pub mod cudnn;
pub mod kernel;
pub mod runtime;

use std::{ptr::NonNull, sync::Mutex};

use cublas::cublas_error::ZenuCublasError;
use once_cell::sync::Lazy;
use runtime::{cuda_create_stream, set_up_mempool};
use zenu_cublas_sys::{cublasContext, cublasCreate_v2, cublasDestroy_v2};
use zenu_cuda_runtime_sys::{cudaMemPool_t, cudaStream_t};
use zenu_cudnn_sys::{cudnnContext, cudnnCreate, cudnnDestroy, cudnnHandle_t};

static ZENU_CUDA_STATE: Lazy<Mutex<ZenuCudaState>> = Lazy::new(|| Mutex::new(ZenuCudaState::new()));

pub struct ZenuCudaState {
    cublas: NonNull<cublasContext>,
    cudnn: NonNull<cudnnContext>,
    stream: cudaStream_t,
    mem_pool: cudaMemPool_t,
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
        let mem_pool = set_up_mempool().unwrap();
        let stream = cuda_create_stream().unwrap();
        Self {
            cublas,
            cudnn,
            stream,
            mem_pool,
        }
    }

    #[must_use]
    pub fn get_cublas(&self) -> NonNull<cublasContext> {
        self.cublas
    }

    #[must_use]
    pub fn get_cudnn(&self) -> NonNull<cudnnContext> {
        self.cudnn
    }

    // pub type cudnnHandle_t = *mut cudnnContext;
    #[must_use]
    pub fn get_cudnn_handle(&self) -> cudnnHandle_t {
        let cudnn_context = self.get_cudnn();
        cudnn_context.as_ptr()
    }

    #[must_use]
    pub fn get_stream(&self) -> cudaStream_t {
        self.stream
    }

    #[must_use]
    pub fn get_mem_pool(&self) -> cudaMemPool_t {
        self.mem_pool
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

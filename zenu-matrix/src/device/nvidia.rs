use super::{Device, DeviceBase};
use crate::{memory_pool::MemPoolError, num::Num, ZENU_MATRIX_STATE};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Nvidia;

impl DeviceBase for Nvidia {
    fn raw_drop_ptr<T>(ptr: *mut T) {
        zenu_cuda::runtime::cuda_free(ptr.cast::<::libc::c_void>()).unwrap();
    }

    fn mem_pool_drop_ptr(ptr: *mut u8) -> Result<(), MemPoolError> {
        let state = &ZENU_MATRIX_STATE;
        state.nvidia.try_free(ptr)
    }

    fn clone_ptr<T>(src: *const T, len: usize) -> *mut T {
        let bytes = len * std::mem::size_of::<T>();
        let dst = Self::alloc(bytes).unwrap().cast::<T>();
        zenu_cuda::runtime::cuda_copy(
            dst,
            src,
            len,
            zenu_cuda::runtime::ZenuCudaMemCopyKind::HostToHost,
        )
        .unwrap();
        dst
    }

    fn assign_item<T: Num>(ptr: *mut T, offset: usize, value: T) {
        zenu_cuda::kernel::set_memory(ptr, offset, value);
    }

    fn get_item<T: Num>(ptr: *const T, offset: usize) -> T {
        zenu_cuda::kernel::get_memory(ptr, offset)
    }

    fn from_vec<T: Num>(mut vec: Vec<T>) -> *mut T {
        let ptr = Self::alloc(vec.len() * std::mem::size_of::<T>())
            .unwrap()
            .cast::<T>();
        zenu_cuda::runtime::cuda_copy(
            ptr,
            vec.as_mut_ptr(),
            vec.len(),
            zenu_cuda::runtime::ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        ptr
    }

    fn zeros<T: Num>(len: usize) -> *mut T {
        let bytes = len * std::mem::size_of::<T>();
        let ptr = Self::alloc(bytes).unwrap().cast::<T>();
        zenu_cuda::cublas::cublas_scal(len, T::zero(), ptr, 1).unwrap();
        ptr
    }

    fn raw_alloc(num_bytes: usize) -> Result<*mut u8, String> {
        zenu_cuda::runtime::cuda_malloc_bytes(num_bytes)
            .map_err(|_| "cudaMalloc failed".to_string())
    }

    fn mem_pool_alloc(num_bytes: usize) -> Result<*mut u8, MemPoolError> {
        let state = &ZENU_MATRIX_STATE;
        state.nvidia.try_alloc(num_bytes)
    }
}

impl Device for Nvidia {}

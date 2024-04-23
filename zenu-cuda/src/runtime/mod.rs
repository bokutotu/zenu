use zenu_cuda_runtime_sys::{cudaError, cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind};

use self::runtime_error::ZenuCudaRuntimeError;

pub mod runtime_error;

pub fn cuda_malloc<T>(size: usize) -> Result<*mut T, ZenuCudaRuntimeError> {
    let mut ptr = std::ptr::null_mut();
    let size = size * std::mem::size_of::<T>();
    let err = unsafe { cudaMalloc(&mut ptr as *mut *mut T as *mut *mut std::ffi::c_void, size) }
        as cudaError as u32;
    let err = ZenuCudaRuntimeError::from(err);
    match err {
        ZenuCudaRuntimeError::CudaSuccess => Ok(ptr),
        _ => Err(err),
    }
}

pub fn cuda_free<T>(ptr: *mut T) -> Result<(), ZenuCudaRuntimeError> {
    let err: ZenuCudaRuntimeError =
        (unsafe { cudaFree(ptr as *mut std::ffi::c_void) } as u32).into();
    match err {
        ZenuCudaRuntimeError::CudaSuccess => Ok(()),
        _ => Err(err),
    }
}

pub enum ZenuCudaMemCopyKind {
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    Default,
}

impl From<ZenuCudaMemCopyKind> for cudaMemcpyKind {
    fn from(value: ZenuCudaMemCopyKind) -> Self {
        match value {
            ZenuCudaMemCopyKind::HostToHost => cudaMemcpyKind::cudaMemcpyHostToHost,
            ZenuCudaMemCopyKind::HostToDevice => cudaMemcpyKind::cudaMemcpyHostToDevice,
            ZenuCudaMemCopyKind::DeviceToHost => cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ZenuCudaMemCopyKind::DeviceToDevice => cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ZenuCudaMemCopyKind::Default => cudaMemcpyKind::cudaMemcpyDefault,
        }
    }
}

pub fn cuda_copy<T>(
    dst: *mut T,
    src: *const T,
    size: usize,
    kind: ZenuCudaMemCopyKind,
) -> Result<(), ZenuCudaRuntimeError> {
    let size = size * std::mem::size_of::<T>();
    let err = unsafe {
        cudaMemcpy(
            dst as *mut std::ffi::c_void,
            src as *const std::ffi::c_void,
            size,
            cudaMemcpyKind::from(kind),
        )
    } as u32;
    let err = ZenuCudaRuntimeError::from(err);
    match err {
        ZenuCudaRuntimeError::CudaSuccess => Ok(()),
        _ => Err(err),
    }
}

pub fn copy_to_gpu<T>(src: *mut T, len: usize) -> *mut T {
    let gpu_ptr = cuda_malloc(len).unwrap();
    cuda_copy(gpu_ptr, src, len, ZenuCudaMemCopyKind::HostToDevice).unwrap();
    gpu_ptr
}

pub fn copy_to_cpu<T: 'static + Default + Clone>(src: *mut T, len: usize) -> *mut T {
    let mut dst = vec![Default::default(); len];
    let dst_ptr = dst.as_mut_ptr();
    cuda_copy(dst_ptr, src, len, ZenuCudaMemCopyKind::DeviceToHost).unwrap();
    std::mem::forget(dst);
    dst_ptr
}

#[cfg(test)]
mod cuda_runtime {
    use crate::runtime::ZenuCudaMemCopyKind;

    use super::{copy_to_cpu, copy_to_gpu, cuda_copy, cuda_malloc};

    #[test]
    fn cpu_to_gpu_to_cpu() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut b = vec![0.0f32; 4];

        let a_ptr = cuda_malloc::<f32>(4).unwrap();
        cuda_copy(a_ptr, a.as_ptr(), 4, ZenuCudaMemCopyKind::HostToDevice).unwrap();

        cuda_copy(b.as_mut_ptr(), a_ptr, 4, ZenuCudaMemCopyKind::DeviceToHost).unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn cpu_to_gou_to_cpu_2() {
        let mut a = vec![1.0f32, 2.0, 3.0, 4.0];

        let gpu_ptr = copy_to_gpu(a.as_mut_ptr(), 4);

        let b_ptr = copy_to_cpu(gpu_ptr, 4);
        let b = unsafe { std::slice::from_raw_parts(b_ptr, 4) };

        assert_eq!(a, b);
    }
}

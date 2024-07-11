use zenu_cuda_runtime_sys::{
    cudaDeviceSetMemPool, cudaError, cudaFreeAsync, cudaMallocAsync, cudaMemAllocationHandleType,
    cudaMemAllocationType, cudaMemGetInfo, cudaMemLocation, cudaMemLocationType, cudaMemPoolAttr,
    cudaMemPoolCreate, cudaMemPoolProps, cudaMemPoolSetAttribute, cudaMemPool_t, cudaMemcpy,
    cudaMemcpyKind, cudaStreamCreate, cudaStreamSynchronize, cudaStream_t, CUmemPoolHandle_st,
};

use crate::ZENU_CUDA_STATE;

use self::runtime_error::ZenuCudaRuntimeError;

pub mod runtime_error;

pub fn cuda_stream_sync() -> Result<(), ZenuCudaRuntimeError> {
    let stream = ZENU_CUDA_STATE.lock().unwrap().get_stream();
    let err = unsafe { cudaStreamSynchronize(stream) } as u32;
    let err = ZenuCudaRuntimeError::from(err);
    match err {
        ZenuCudaRuntimeError::CudaSuccess => Ok(()),
        _ => Err(err),
    }
}

pub fn cuda_malloc<T>(size: usize) -> Result<*mut T, ZenuCudaRuntimeError> {
    let mut ptr = std::ptr::null_mut();
    let size = size * std::mem::size_of::<T>();
    let stream = ZENU_CUDA_STATE.lock().unwrap().get_stream();
    let err = unsafe {
        cudaMallocAsync(
            &mut ptr as *mut *mut T as *mut *mut std::ffi::c_void,
            size,
            stream,
        )
    } as cudaError as u32;
    cuda_stream_sync()?;
    let err = ZenuCudaRuntimeError::from(err);
    match err {
        ZenuCudaRuntimeError::CudaSuccess => Ok(ptr),
        _ => Err(err),
    }
}

pub fn cuda_free<T>(ptr: *mut T) -> Result<(), ZenuCudaRuntimeError> {
    let stream = ZENU_CUDA_STATE.lock().unwrap().get_stream();
    let err = unsafe { cudaFreeAsync(ptr as *mut std::ffi::c_void, stream) } as u32;
    let err = ZenuCudaRuntimeError::from(err);
    cuda_stream_sync()?;
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

pub fn cuda_create_stream() -> Result<cudaStream_t, ZenuCudaRuntimeError> {
    let mut stream = std::ptr::null_mut();
    let err = unsafe { cudaStreamCreate(&mut stream) } as u32;
    let err = ZenuCudaRuntimeError::from(err);
    match err {
        ZenuCudaRuntimeError::CudaSuccess => Ok(stream as cudaStream_t),
        _ => Err(err),
    }
}

pub fn cuda_create_pool_props() -> cudaMemPoolProps {
    let mut props = cudaMemPoolProps::default();
    props.allocType = cudaMemAllocationType::cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemAllocationHandleType::cudaMemHandleTypeNone;
    let mut cuda_location = cudaMemLocation::default();
    cuda_location.type_ = cudaMemLocationType::cudaMemLocationTypeDevice;
    cuda_location.id = 0;
    props.location = cuda_location;
    props
}

pub struct MemoryInfo {
    pub free: usize,
    pub total: usize,
}

pub fn cuda_get_memory_info() -> Result<MemoryInfo, ZenuCudaRuntimeError> {
    let mut free = 0;
    let mut total = 0;
    let err = unsafe { cudaMemGetInfo(&mut free as *mut usize, &mut total as *mut usize) } as u32;
    let err = ZenuCudaRuntimeError::from(err);
    match err {
        ZenuCudaRuntimeError::CudaSuccess => Ok(MemoryInfo { free, total }),
        _ => Err(err),
    }
}

pub fn cuda_set_mem_pool(
    dev_id: usize,
    mempool: cudaMemPool_t,
) -> Result<(), ZenuCudaRuntimeError> {
    let dev_id = dev_id as ::libc::c_int;
    let err = unsafe { cudaDeviceSetMemPool(dev_id, mempool) } as u32;
    let err = ZenuCudaRuntimeError::from(err);
    match err {
        ZenuCudaRuntimeError::CudaSuccess => Ok(()),
        _ => Err(err),
    }
}

pub fn cuda_set_mem_pool_atribute_mem_max(
    mempool: cudaMemPool_t,
    poolsize: usize,
) -> Result<(), ZenuCudaRuntimeError> {
    let poolsize = poolsize as ::libc::size_t;
    let err = unsafe {
        cudaMemPoolSetAttribute(
            mempool,
            cudaMemPoolAttr::cudaMemPoolAttrReleaseThreshold,
            &poolsize as *const ::libc::size_t as *mut std::ffi::c_void,
        )
    } as u32;
    let err = ZenuCudaRuntimeError::from(err);
    match err {
        ZenuCudaRuntimeError::CudaSuccess => Ok(()),
        _ => Err(err),
    }
}

pub fn cuda_create_mem_pool() -> Result<cudaMemPool_t, ZenuCudaRuntimeError> {
    let props = cuda_create_pool_props();
    let mut addr_of_cumempoolhandle: *mut CUmemPoolHandle_st = std::ptr::null_mut();
    let mempool_ptr = &mut addr_of_cumempoolhandle as *mut *mut CUmemPoolHandle_st;
    let err = unsafe {
        cudaMemPoolCreate(
            mempool_ptr as *mut cudaMemPool_t,
            &props as *const cudaMemPoolProps,
        )
    } as u32;
    match err {
        0 => Ok(unsafe { *mempool_ptr }),
        _ => Err(ZenuCudaRuntimeError::from(err as u32)),
    }
}

pub fn set_up_mempool() -> Result<cudaMemPool_t, ZenuCudaRuntimeError> {
    let mempool = cuda_create_mem_pool()?;
    cuda_set_mem_pool(0, mempool)?;
    let mem_info = cuda_get_memory_info()?;
    let pool_size = ((mem_info.free as f32) * 0.99) as usize;
    cuda_set_mem_pool_atribute_mem_max(mempool, pool_size)?;
    Ok(mempool)
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

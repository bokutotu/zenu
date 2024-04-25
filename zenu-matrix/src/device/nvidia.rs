use super::DeviceBase;
use crate::num::Num;

#[derive(Copy, Clone, Default)]
pub struct Nvidia;

impl DeviceBase for Nvidia {
    fn drop_ptr<T>(ptr: *mut T, _: usize) {
        zenu_cuda::runtime::cuda_free(ptr as *mut std::ffi::c_void).unwrap();
    }

    fn clone_ptr<T>(src: *const T, len: usize) -> *mut T {
        let dst = zenu_cuda::runtime::cuda_malloc(len).unwrap() as *mut T;
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
        zenu_cuda::runtime::copy_to_gpu(vec.as_mut_ptr(), vec.len())
    }
}

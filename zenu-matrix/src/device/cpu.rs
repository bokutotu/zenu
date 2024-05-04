use super::DeviceBase;

#[derive(Copy, Clone, Default)]
pub struct Cpu;

impl DeviceBase for Cpu {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn drop_ptr<T>(ptr: *mut T, len: usize) {
        unsafe {
            std::vec::Vec::from_raw_parts(ptr, 0, len);
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn clone_ptr<T>(ptr: *const T, len: usize) -> *mut T {
        let mut vec = Vec::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), len);
        }
        vec.as_mut_ptr()
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn assign_item<T>(ptr: *mut T, offset: usize, value: T) {
        unsafe {
            ptr.add(offset).write(value);
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn get_item<T>(ptr: *const T, offset: usize) -> T {
        unsafe { ptr.add(offset).read() }
    }

    fn from_vec<T>(vec: Vec<T>) -> *mut T {
        let ptr = vec.as_ptr() as *mut T;
        std::mem::forget(vec);
        ptr
    }
}

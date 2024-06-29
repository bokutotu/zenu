use super::{Device, DeviceBase};

#[derive(Copy, Clone, Default, Debug)]
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
        for i in 0..len {
            vec.push(unsafe { ptr.offset(i as isize).read() });
        }
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        ptr
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

impl Device for Cpu {}

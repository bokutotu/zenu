use super::Device;

#[derive(Copy, Clone, Default)]
pub struct Cpu;

impl Device for Cpu {
    fn drop_ptr<T>(ptr: *mut T, len: usize) {
        unsafe {
            std::vec::Vec::from_raw_parts(ptr, 0, len);
        }
    }

    fn clone_ptr<T>(ptr: *const T, len: usize) -> *mut T {
        let mut vec = Vec::with_capacity(len);
        unsafe {
            vec.set_len(len);
            std::ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), len);
        }
        vec.as_mut_ptr()
    }

    fn assign_item<T>(ptr: *mut T, offset: usize, value: T) {
        unsafe {
            ptr.add(offset).write(value);
        }
    }

    fn get_item<T>(ptr: *const T, offset: usize) -> T {
        unsafe { ptr.add(offset).read() }
    }
}

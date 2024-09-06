use serde::{Deserialize, Serialize};

use crate::{memory_pool::MemPoolError, num::Num, ZENU_MATRIX_STATE};

use super::{Device, DeviceBase};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Cpu;

impl DeviceBase for Cpu {
    fn raw_drop_ptr<T>(ptr: *mut T) {
        unsafe { libc::free(ptr.cast::<::libc::c_void>()) }
    }

    fn mem_pool_drop_ptr(ptr: *mut u8) -> Result<(), MemPoolError> {
        let state = &ZENU_MATRIX_STATE;
        state.cpu.try_free(ptr)
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn clone_ptr<T>(ptr: *const T, len: usize) -> *mut T {
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            vec.push(unsafe { ptr.add(i).read() });
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

    fn from_vec<T>(mut vec: Vec<T>) -> *mut T {
        let ptr = vec.as_mut_ptr().cast::<T>();
        std::mem::forget(vec);
        ptr
    }

    fn zeros<T: Num>(len: usize) -> *mut T {
        use cblas::{dscal, sscal};
        let ptr = Self::alloc(len * std::mem::size_of::<T>())
            .unwrap()
            .cast::<T>();
        if T::is_f32() {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr.cast(), 1) };
            unsafe { sscal(i32::try_from(len).unwrap(), 0.0, slice, 1) };
        } else {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr.cast(), 1) };
            unsafe { dscal(i32::try_from(len).unwrap(), 0.0, slice, 1) };
        }
        ptr
    }

    fn raw_alloc(num_bytes: usize) -> Result<*mut u8, String> {
        let ptr = unsafe { libc::malloc(num_bytes) };
        if ptr.is_null() {
            Err("null pointer".to_string())
        } else {
            Ok(ptr.cast())
        }
    }

    fn mem_pool_alloc(num_bytes: usize) -> Result<*mut u8, MemPoolError> {
        let state = &ZENU_MATRIX_STATE;
        state.cpu.try_alloc(num_bytes)
    }
}

impl Device for Cpu {}

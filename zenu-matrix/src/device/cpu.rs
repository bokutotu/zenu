use serde::{Deserialize, Serialize};

use crate::{num::Num, ZENU_MATRIX_STATE};

use super::{Device, DeviceBase};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Cpu;

impl DeviceBase for Cpu {
    fn raw_drop_ptr<T>(ptr: *mut T) {
        unsafe { libc::free(ptr as *mut libc::c_void) }
    }

    fn mem_pool_drop_ptr(ptr: *mut u8) -> Result<(), ()> {
        let state = &ZENU_MATRIX_STATE;
        state.cpu_mem_pool.try_free(ptr)
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

    fn from_vec<T>(vec: Vec<T>) -> *mut T {
        let ptr = vec.as_ptr() as *mut T;
        std::mem::forget(vec);
        ptr
    }

    fn zeros<T: Num>(len: usize) -> *mut T {
        use cblas::*;
        let ptr = Self::alloc(len * std::mem::size_of::<T>()).unwrap() as *mut T;
        println!("ptr deref: {:?}", unsafe { *ptr });
        if T::is_f32() {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, 1) };
            unsafe { sscal(len as i32, 0.0, slice, 1) };
        } else {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f64, 1) };
            unsafe { dscal(len as i32, 0.0, slice, 1) };
        }
        ptr
    }

    fn raw_alloc(num_bytes: usize) -> Result<*mut u8, ()> {
        let ptr = unsafe { libc::malloc(num_bytes) };
        if ptr.is_null() {
            Err(())
        } else {
            Ok(ptr as *mut u8)
        }
    }

    fn mem_pool_alloc(num_bytes: usize) -> Result<*mut u8, ()> {
        let state = &ZENU_MATRIX_STATE;
        state.cpu_mem_pool.try_alloc(num_bytes)
    }
}

impl Device for Cpu {}

use std::any::TypeId;

use serde::{Deserialize, Serialize};

use crate::num::Num;

use super::{Device, DeviceBase};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Cpu;

impl DeviceBase for Cpu {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn drop_ptr<T>(ptr: *mut T, len: usize) {
        unsafe {
            let _ = std::vec::Vec::from_raw_parts(ptr, len, len);
        }
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
        let mut vec = Vec::with_capacity(len);
        unsafe { vec.set_len(len) };
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        if T::is_f32() {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, len) };
            unsafe { sscal(len as i32, 0.0, slice, 1) };
        } else {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f64, len) };
            unsafe { dscal(len as i32, 0.0, slice, 1) };
        }
        ptr
    }
}

impl Device for Cpu {}

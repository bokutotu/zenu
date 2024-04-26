use crate::num::Num;

pub mod cpu;

#[cfg(feature = "nvidia")]
pub mod nvidia;

pub trait DeviceBase: Copy + Default + 'static {
    fn drop_ptr<T>(ptr: *mut T, len: usize);
    fn clone_ptr<T>(ptr: *const T, len: usize) -> *mut T;
    fn assign_item<T: Num>(ptr: *mut T, offset: usize, value: T);
    fn get_item<T: Num>(ptr: *const T, offset: usize) -> T;
    fn from_vec<T: Num>(vec: Vec<T>) -> *mut T;
}

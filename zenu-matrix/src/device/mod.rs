use serde::Serialize;

use crate::{
    memory_pool::MemPoolError,
    nn::{batch_norm::BatchNormalization, conv2d::Conv2d, pool2d::Pool2dImpl},
    num::Num,
    operation::{
        asum::Asum,
        basic_operations::{
            AbsOps, AcosOps, AddOps, AsinOps, AtanOps, CosOps, CoshOps, DivOps, ExpOps, LogOps,
            MulOps, PowOws, SinOps, SinhOps, SqrtOps, SubOps, TanOps, TanhOps,
        },
        clip::ClipOps,
        copy_from::CopyBlas,
        max::MaxIdx,
        mul::Gemm,
        relu::ReluOps,
    },
    ZENU_MATRIX_STATE,
};

pub mod cpu;

#[cfg(feature = "nvidia")]
pub mod nvidia;

pub trait DeviceBase: Copy + Default + Serialize + 'static {
    fn drop_ptr<T>(ptr: *mut T) {
        let state = &ZENU_MATRIX_STATE;
        if state.use_mem_pool {
            let result = Self::mem_pool_drop_ptr(ptr as *mut u8);
            if result.is_err() {
                Self::raw_drop_ptr(ptr)
            }
        } else {
            Self::raw_drop_ptr(ptr)
        }
    }
    fn mem_pool_drop_ptr(ptr: *mut u8) -> Result<(), MemPoolError>;
    fn raw_drop_ptr<T>(ptr: *mut T);
    fn clone_ptr<T>(ptr: *const T, len: usize) -> *mut T;
    fn assign_item<T: Num>(ptr: *mut T, offset: usize, value: T);
    fn get_item<T: Num>(ptr: *const T, offset: usize) -> T;
    fn from_vec<T: Num>(vec: Vec<T>) -> *mut T;
    fn zeros<T: Num>(len: usize) -> *mut T;
    fn alloc(num_bytes: usize) -> Result<*mut u8, MemPoolError> {
        let state = &ZENU_MATRIX_STATE;
        if state.use_mem_pool {
            Self::mem_pool_alloc(num_bytes)
        } else {
            Self::raw_alloc(num_bytes).map_err(|_| MemPoolError::DeviceMallocError)
        }
    }
    fn mem_pool_alloc(num_bytes: usize) -> Result<*mut u8, MemPoolError>;
    fn raw_alloc(num_bytes: usize) -> Result<*mut u8, String>;
}

pub trait Device:
    DeviceBase
    + CopyBlas
    + AddOps
    + SubOps
    + MulOps
    + DivOps
    + Asum
    + ClipOps
    + SinOps
    + CosOps
    + TanOps
    + AsinOps
    + AcosOps
    + AtanOps
    + SinhOps
    + CoshOps
    + TanhOps
    + AbsOps
    + SqrtOps
    + ExpOps
    + LogOps
    + MaxIdx
    + ReluOps
    + Gemm
    + PowOws
    + BatchNormalization
    + Conv2d
    + Sized
    + Pool2dImpl
    + Send
    + Sync
    + 'static
{
}

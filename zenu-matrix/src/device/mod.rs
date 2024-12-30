use serde::Serialize;

use crate::{
    memory_pool::MemPoolError,
    nn::{
        batch_norm::BatchNormalization,
        conv::interface::{ConvBkwdData, ConvBkwdFilter, ConvFwd},
        conv2d::Conv2d,
        dropout::Dropout,
        pool2d::Pool2dImpl,
    },
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

#[expect(clippy::module_name_repetitions)]
pub trait DeviceBase: Copy + Default + Serialize + 'static {
    fn drop_ptr<T>(ptr: *mut T) {
        let state = &ZENU_MATRIX_STATE;
        if state.is_mem_pool_used {
            let result = Self::mem_pool_drop_ptr(ptr.cast());
            if result.is_err() {
                Self::raw_drop_ptr(ptr);
            }
        } else {
            Self::raw_drop_ptr(ptr);
        }
    }
    #[expect(clippy::missing_errors_doc)]
    fn mem_pool_drop_ptr(ptr: *mut u8) -> Result<(), MemPoolError>;
    fn raw_drop_ptr<T>(ptr: *mut T);
    fn clone_ptr<T>(ptr: *const T, len: usize) -> *mut T;
    fn assign_item<T: Num>(ptr: *mut T, offset: usize, value: T);
    fn get_item<T: Num>(ptr: *const T, offset: usize) -> T;
    fn from_vec<T: Num>(vec: Vec<T>) -> *mut T;
    fn zeros<T: Num>(len: usize) -> *mut T;
    #[expect(clippy::missing_errors_doc)]
    fn alloc(num_bytes: usize) -> Result<*mut u8, MemPoolError> {
        if num_bytes == 0 {
            return Ok(std::ptr::null_mut());
        }
        let state = &ZENU_MATRIX_STATE;
        if state.is_mem_pool_used {
            Self::mem_pool_alloc(num_bytes)
        } else {
            Self::raw_alloc(num_bytes).map_err(|_| MemPoolError::DeviceMallocError)
        }
    }
    #[expect(clippy::missing_errors_doc)]
    fn mem_pool_alloc(num_bytes: usize) -> Result<*mut u8, MemPoolError>;
    #[expect(clippy::missing_errors_doc)]
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
    + ConvFwd
    + ConvBkwdData
    + ConvBkwdFilter
    + Sized
    + Pool2dImpl
    + Dropout
    + Send
    + Sync
    + 'static
{
}

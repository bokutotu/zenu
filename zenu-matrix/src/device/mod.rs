use serde::{Deserialize, Serialize};

use crate::{
    nn::{batch_norm::BatchNormalization, conv2d::Conv2d},
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
};

pub mod cpu;

#[cfg(feature = "nvidia")]
pub mod nvidia;

pub trait DeviceBase: Copy + Default + Serialize + for<'de> Deserialize<'de> + 'static {
    fn drop_ptr<T>(ptr: *mut T, len: usize);
    fn clone_ptr<T>(ptr: *const T, len: usize) -> *mut T;
    fn assign_item<T: Num>(ptr: *mut T, offset: usize, value: T);
    fn get_item<T: Num>(ptr: *const T, offset: usize) -> T;
    fn from_vec<T: Num>(vec: Vec<T>) -> *mut T;
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
    + 'static
{
}

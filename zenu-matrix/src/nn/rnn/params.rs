use crate::{
    device::{nvidia::Nvidia, DeviceBase},
    dim::DimDyn,
    matrix::{Matrix, Owned, Ref},
    num::Num,
};

pub struct RNNOutput<T: Num> {
    pub y: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub hy: Matrix<Owned<T>, DimDyn, Nvidia>,
}

pub struct RNNBkwdDataOutput<T: Num> {
    pub dx: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub dhx: Matrix<Owned<T>, DimDyn, Nvidia>,
}

pub struct RNNParameters {
    pub weight: *mut u8,
}

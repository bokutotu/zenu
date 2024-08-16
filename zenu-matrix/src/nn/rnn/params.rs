use crate::{
    device::DeviceBase,
    dim::DimDyn,
    matrix::{Matrix, Owned, Ref},
    num::Num,
};

pub struct RNNOutput<T: Num, D: DeviceBase> {
    pub y: Matrix<Owned<T>, DimDyn, D>,
    pub hy: Matrix<Owned<T>, DimDyn, D>,
}

pub struct RNNBkwdDataOutput<T: Num, D: DeviceBase> {
    pub dx: Matrix<Owned<T>, DimDyn, D>,
    pub dhx: Matrix<Owned<T>, DimDyn, D>,
}

pub struct RNNBkwdWeightsOutputCpu<T: Num, D: DeviceBase> {
    pub dwx: Matrix<Owned<T>, DimDyn, D>,
    pub dwh: Matrix<Owned<T>, DimDyn, D>,
    pub db: Matrix<Owned<T>, DimDyn, D>,
}

pub struct RNNBackwardWeightsOutputNvidia {
    pub dw: *mut u8,
}

pub enum RNNBkwdWeightsOutput<T: Num, D: DeviceBase> {
    Cpu(Vec<RNNBkwdWeightsOutputCpu<T, D>>),
    #[cfg(feature = "nvidia")]
    Nvidia(RNNBackwardWeightsOutputNvidia),
}

pub struct RNNParametersCpu<'a, T: Num, D: DeviceBase> {
    pub input_weight: Matrix<Ref<&'a T>, DimDyn, D>,
    pub input_bias: Matrix<Ref<&'a T>, DimDyn, D>,
    pub hidden_weight: Matrix<Ref<&'a T>, DimDyn, D>,
    pub hidden_bias: Matrix<Ref<&'a T>, DimDyn, D>,
}

pub struct RNNParametersNvidia {
    pub weight: *mut u8,
}

pub enum RNNParameters<'a, T: Num, D: DeviceBase> {
    Cpu(RNNParametersCpu<'a, T, D>),
    #[cfg(feature = "nvidia")]
    Nvidia(RNNParametersNvidia),
}

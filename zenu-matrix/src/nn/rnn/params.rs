use crate::{
    device::DeviceBase,
    dim::DimDyn,
    matrix::{Matrix, Ref},
    num::Num,
};

pub struct RNNOutput<'a, T: Num, D: DeviceBase> {
    pub x: Matrix<Ref<&'a T>, DimDyn, D>,
    pub hy: Matrix<Ref<&'a T>, DimDyn, D>,
}

pub struct RNNBkwdDataOutput<'a, T: Num, D: DeviceBase> {
    pub dx: Matrix<Ref<&'a T>, DimDyn, D>,
    pub dhx: Matrix<Ref<&'a T>, DimDyn, D>,
}

pub struct RNNBkwdWeightsOutput<'a, T: Num, D: DeviceBase> {
    pub dwx: Matrix<Ref<&'a T>, DimDyn, D>,
    pub dwh: Matrix<Ref<&'a T>, DimDyn, D>,
    pub db: Matrix<Ref<&'a T>, DimDyn, D>,
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

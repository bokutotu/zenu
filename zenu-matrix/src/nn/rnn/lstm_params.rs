use crate::{
    device::nvidia::Nvidia,
    dim::DimDyn,
    matrix::{Matrix, Owned},
    num::Num,
};

#[derive(Debug, Clone)]
pub struct LSTMOutput<T: Num> {
    pub y: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub hy: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub cy: Matrix<Owned<T>, DimDyn, Nvidia>,
}

#[derive(Debug, Clone)]
pub struct LSTMGrad<T: Num> {
    pub dx: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub dhx: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub dcx: Matrix<Owned<T>, DimDyn, Nvidia>,
}

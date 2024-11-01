use crate::{
    device::nvidia::Nvidia,
    dim::DimDyn,
    matrix::{Matrix, Owned},
    num::Num,
};

#[derive(Debug, Clone)]
pub struct GRUOutput<T: Num> {
    pub y: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub hy: Matrix<Owned<T>, DimDyn, Nvidia>,
}

#[derive(Debug, Clone)]
pub struct GRUGrad<T: Num> {
    pub dx: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub dhx: Matrix<Owned<T>, DimDyn, Nvidia>,
}

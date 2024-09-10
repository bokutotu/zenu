use zenu_matrix::{device::Device, num::Num};

use crate::Variable;

pub mod cudnn;
pub mod naive;

pub struct RNNOutput<T: Num, D: Device> {
    pub y: Variable<T, D>,
    pub hy: Variable<T, D>,
}

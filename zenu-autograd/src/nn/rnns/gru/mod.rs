use crate::Variable;

use zenu_matrix::{device::Device, num::Num};

#[cfg(feature = "nvidia")]
pub mod cudnn;

pub mod naive;

pub struct GRUOutput<T: Num, D: Device> {
    pub y: Variable<T, D>,
    pub hy: Variable<T, D>,
}

pub mod adam;
// pub mod adamw;
pub mod sgd;

use zenu_autograd::Variable;
use zenu_matrix::{device::Device, num::Num};

pub trait Optimizer<T: Num, D: Device> {
    fn update(&self, parameters: &[Variable<T, D>]);
}

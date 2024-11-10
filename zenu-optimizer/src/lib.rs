pub mod adam;
pub mod adamw;
pub mod sgd;

use zenu_layer::Parameters;
use zenu_matrix::{device::Device, num::Num};

pub trait Optimizer<T: Num, D: Device, P: Parameters<T, D>> {
    fn update(&self, parameters: &P);
}

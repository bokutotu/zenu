use zenu_autograd::Variable;
use zenu_matrix::{device::Device, num::Num};

pub mod layers;

pub trait Module<T: Num, D: Device> {
    fn call(&self, input: Variable<T, D>) -> Variable<T, D>;
}

use serde::{Deserialize, Serialize};
use zenu_autograd::Variable;
use zenu_matrix::{device::Device, num::Num};

pub mod layers;

pub trait Module<T: Num, D: Device> {
    fn call(&self, input: Variable<T, D>) -> Variable<T, D>;
}

pub trait Parameters<T: Num, D: Device> {
    fn weights(&self) -> Vec<&Variable<T, D>>;
    fn biases(&self) -> Vec<&Variable<T, D>>;
    fn parameters(&self) -> Vec<&Variable<T, D>> {
        let mut params = self.weights();
        params.extend(self.biases());
        params
    }
}

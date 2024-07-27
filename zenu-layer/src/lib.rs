use std::collections::HashMap;

use zenu_autograd::Variable;
use zenu_matrix::{device::Device, num::Num};

pub mod layers;

pub trait Module<T: Num, D: Device> {
    fn call(&self, input: Variable<T, D>) -> Variable<T, D>;
}

pub trait Parameters<T: Num, D: Device> {
    fn weights(&self) -> HashMap<String, Variable<T, D>>;
    fn biases(&self) -> HashMap<String, Variable<T, D>>;
    fn parameters(&self) -> HashMap<String, Variable<T, D>> {
        let weights = self.weights();
        let biases = self.biases();
        let mut parameters = HashMap::new();
        for (key, value) in weights.iter() {
            parameters.insert(key.clone(), value.clone());
        }
        for (key, value) in biases.iter() {
            parameters.insert(key.clone(), value.clone());
        }
        parameters
    }
}

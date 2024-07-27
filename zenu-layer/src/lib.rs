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

impl<T: Num, D: Device> Parameters<T, D> for () {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        HashMap::new()
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        HashMap::new()
    }
}

impl<T: Num, D: Device> Parameters<T, D> for Vec<Box<dyn Parameters<T, D>>> {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        let mut weights = HashMap::new();
        for (idx, param) in self.iter().enumerate() {
            for (key, value) in param.weights().iter() {
                weights.insert(format!("{}.{}", idx, key), value.clone());
            }
        }
        weights
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        let mut biases = HashMap::new();
        for (idx, param) in self.iter().enumerate() {
            for (key, value) in param.biases().iter() {
                biases.insert(format!("{}.{}", idx, key), value.clone());
            }
        }
        biases
    }
}

impl<T: Num, D: Device> Parameters<T, D> for Box<dyn Parameters<T, D>> {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        self.as_ref().weights()
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        self.as_ref().biases()
    }
}

impl<T: Num, D: Device> Parameters<T, D> for HashMap<String, Box<dyn Parameters<T, D>>> {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        let mut weights = HashMap::new();
        for (key, param) in self.iter() {
            for (sub_key, value) in param.weights().iter() {
                weights.insert(format!("{}.{}", key, sub_key), value.clone());
            }
        }
        weights
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        let mut biases = HashMap::new();
        for (key, param) in self.iter() {
            for (sub_key, value) in param.biases().iter() {
                biases.insert(format!("{}.{}", key, sub_key), value.clone());
            }
        }
        biases
    }
}

use std::collections::HashMap;

use zenu_autograd::Variable;
use zenu_matrix::{device::Device, num::Num};

pub mod layers;

pub trait ModuleParameters<T: Num, D: Device> {}

impl<T: Num, D: Device> ModuleParameters<T, D> for () {}

impl<T: Num, D: Device> ModuleParameters<T, D> for Variable<T, D> {}

impl<T: Num, D: Device> ModuleParameters<T, D> for Vec<Variable<T, D>> {}

impl<T: Num, D: Device, K, S: ::std::hash::BuildHasher> ModuleParameters<T, D>
    for HashMap<K, Variable<T, D>, S>
{
}

pub trait Module<T: Num, D: Device> {
    type Input: ModuleParameters<T, D>;
    type Output: ModuleParameters<T, D>;
    fn call(&self, input: Self::Input) -> Self::Output;
}

pub trait Parameters<T: Num, D: Device> {
    fn weights(&self) -> HashMap<String, Variable<T, D>>;
    fn biases(&self) -> HashMap<String, Variable<T, D>>;
    fn parameters(&self) -> HashMap<String, Variable<T, D>> {
        let weights = self.weights();
        let biases = self.biases();
        let mut parameters = HashMap::new();
        for (key, value) in weights {
            parameters.insert(key.clone(), value.clone());
        }
        for (key, value) in biases {
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
impl<T: Num, D: Device, P: Parameters<T, D>> Parameters<T, D> for Vec<P> {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        let mut weights = HashMap::new();
        for (idx, param) in self.iter().enumerate() {
            for (key, value) in param.weights() {
                weights.insert(format!("{idx}.{key}"), value.clone());
            }
        }
        weights
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        let mut biases = HashMap::new();
        for (idx, param) in self.iter().enumerate() {
            for (key, value) in param.biases() {
                biases.insert(format!("{idx}.{key}",), value.clone());
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

impl<T: Num, D: Device, P: Parameters<T, D>, S: ::std::hash::BuildHasher> Parameters<T, D>
    for HashMap<String, P, S>
{
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        let mut weights = HashMap::new();
        for (key, param) in self {
            for (sub_key, value) in param.weights() {
                weights.insert(format!("{key}.{sub_key}"), value.clone());
            }
        }
        weights
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        let mut biases = HashMap::new();
        for (key, param) in self {
            for (sub_key, value) in param.biases() {
                biases.insert(format!("{key}.{sub_key}"), value.clone());
            }
        }
        biases
    }
}

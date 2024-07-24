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

pub trait StateDict<'de>: Serialize + Deserialize<'de> {
    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    #[allow(clippy::must_use_candidate)]
    fn from_json(json: &'de str) -> Self {
        serde_json::from_str(json).unwrap()
    }

    fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap()
    }

    #[allow(clippy::must_use_candidate)]
    fn from_bytes(bytes: &'de [u8]) -> Self {
        bincode::deserialize(bytes).unwrap()
    }
}

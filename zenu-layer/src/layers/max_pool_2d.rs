use serde::{Deserialize, Serialize};
use zenu_autograd::{
    functions::pool2d::{max_pool_2d, MaxPool2dConfig},
    Variable,
};
use zenu_matrix::{device::Device, num::Num};

use crate::{Module, Parameters};

#[derive(Serialize, Deserialize)]
#[serde(bound(deserialize = "T: Num + Deserialize<'de>"))]
pub struct MaxPool2d<T: Num> {
    stride: (usize, usize),
    kernel_size: (usize, usize),
    pad: (usize, usize),
    #[serde(skip)]
    config: MaxPool2dConfig<T>,
}

impl<T: Num, D: Device> Parameters<T, D> for MaxPool2d<T> {
    fn weights(&self) -> Vec<&Variable<T, D>> {
        Vec::new()
    }

    fn biases(&self) -> Vec<&Variable<T, D>> {
        Vec::new()
    }
}

impl<T: Num> MaxPool2d<T> {
    #[must_use]
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize), pad: (usize, usize)) -> Self {
        Self {
            stride,
            kernel_size,
            pad,
            config: MaxPool2dConfig::default(),
        }
    }
}

impl<T: Num, D: Device> Module<T, D> for MaxPool2d<T> {
    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
        max_pool_2d(
            input,
            self.kernel_size,
            self.stride,
            self.pad,
            self.config.clone(),
        )
    }
}

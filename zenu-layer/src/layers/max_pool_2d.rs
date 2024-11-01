use std::collections::HashMap;

use zenu_autograd::{
    nn::pool2d::{max_pool_2d, MaxPool2dConfig},
    Variable,
};
use zenu_matrix::{device::Device, num::Num};

use crate::{Module, Parameters};

pub struct MaxPool2d<T: Num> {
    stride: (usize, usize),
    kernel_size: (usize, usize),
    pad: (usize, usize),
    config: MaxPool2dConfig<T>,
}

impl<T: Num, D: Device> Parameters<T, D> for MaxPool2d<T> {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        HashMap::new()
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        HashMap::new()
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
    type Input = Variable<T, D>;
    type Output = Variable<T, D>;
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

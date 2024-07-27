use std::collections::HashMap;

use zenu_autograd::{
    creator::{ones::ones, zeros::zeros},
    functions::batch_norm::{batch_norm_2d, BatchNorm2dAutoGradConfig},
    Variable,
};
use zenu_matrix::{device::Device, dim::DimTrait, num::Num};

use crate::{Module, Parameters};

pub struct BatchNorm2d<T: Num, D: Device> {
    config: BatchNorm2dAutoGradConfig<T>,
    momentum: f64,
    pub scale: Variable<T, D>,
    pub bias: Variable<T, D>,
    pub mean: Variable<T, D>,
    pub variance: Variable<T, D>,
}

impl<T: Num, D: Device> Module<T, D> for BatchNorm2d<T, D> {
    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
        if input.get_shape() != self.config.get_shape() {
            self.config.update_shape(input.get_shape().slice());
        }
        batch_norm_2d(
            input,
            self.scale.clone(),
            self.bias.clone(),
            self.mean.clone(),
            self.variance.clone(),
            self.momentum,
            self.config.clone(),
        )
    }
}

impl<T: Num, D: Device> Parameters<T, D> for BatchNorm2d<T, D> {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        let mut weights = HashMap::new();
        weights.insert("batch_norm_2d.scale".to_string(), self.scale.clone());
        weights
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        let mut biases = HashMap::new();
        biases.insert("batch_norm_2d.bias".to_string(), self.bias.clone());
        biases
    }

    fn parameters(&self) -> HashMap<String, Variable<T, D>> {
        let mut parameters = HashMap::new();
        for (key, value) in self.weights().iter() {
            parameters.insert(key.clone(), value.clone());
        }
        for (key, value) in self.biases().iter() {
            parameters.insert(key.clone(), value.clone());
        }
        parameters.insert("batch_norm_2d.mean".to_string(), self.mean.clone());
        parameters.insert("batch_norm_2d.variance".to_string(), self.variance.clone());
        parameters
    }
}

impl<T: Num, D: Device> BatchNorm2d<T, D> {
    #[must_use]
    pub fn new(channels: usize, momentum: f64) -> Self {
        let scale = ones([channels]);
        let bias = zeros([channels]);
        let mean = zeros([channels]);
        let variance = ones([channels]);

        scale.set_is_train(true);
        bias.set_is_train(true);

        scale.set_name("batch_norm_2d.scale");
        bias.set_name("batch_norm_2d.bias");
        mean.set_name("batch_norm_2d.mean");
        variance.set_name("batch_norm_2d.variance");

        let config = BatchNorm2dAutoGradConfig::default();
        Self {
            config,
            momentum,
            scale,
            bias,
            mean,
            variance,
        }
    }

    pub fn to<Dout: Device>(self) -> BatchNorm2d<T, Dout> {
        BatchNorm2d {
            config: self.config,
            momentum: self.momentum,
            scale: self.scale.to(),
            bias: self.bias.to(),
            mean: self.mean.to(),
            variance: self.variance.to(),
        }
    }
}

use serde::{Deserialize, Serialize};
use zenu_autograd::{
    creator::{ones::ones, zeros::zeros},
    functions::batch_norm::{batch_norm_2d, BatchNorm2dAutoGradConfig},
    Variable,
};
use zenu_matrix::{device::Device, dim::DimTrait, num::Num};

use crate::{Module, Parameteres, StateDict};

#[derive(Serialize, Deserialize)]
#[serde(bound(deserialize = "T: Num + Deserialize<'de>"))]
pub struct BatchNorm2d<T: Num, D: Device> {
    #[serde(skip)]
    config: BatchNorm2dAutoGradConfig<T>,
    momentum: f64,
    pub scale: Variable<T, D>,
    pub bias: Variable<T, D>,
    pub mean: Variable<T, D>,
    pub variance: Variable<T, D>,
}

impl<'de, T: Num + Deserialize<'de>, D: Device> StateDict<'de> for BatchNorm2d<T, D> {}

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

impl<T: Num, D: Device> Parameteres<T, D> for BatchNorm2d<T, D> {
    fn weights(&self) -> Vec<&Variable<T, D>> {
        vec![&self.scale]
    }

    fn biases(&self) -> Vec<&Variable<T, D>> {
        vec![&self.bias]
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

use zenu_autograd::{
    creator::{ones::ones, zeros::zeros},
    functions::batch_norm::{batch_norm_2d, BatchNorm2dAutoGradConfig},
    Variable,
};
use zenu_matrix::{
    device::Device,
    dim::{DimDyn, DimTrait},
    num::Num,
};

use crate::Layer;

pub struct BatchNorm2d<T: Num, D: Device> {
    config: BatchNorm2dAutoGradConfig<T>,
    momentum: f64,
    scale: Variable<T, D>,
    bias: Variable<T, D>,
    mean: Variable<T, D>,
    variance: Variable<T, D>,
}

impl<T: Num, D: Device> Layer<T, D> for BatchNorm2d<T, D> {
    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
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

    fn parameters(&self) -> Vec<Variable<T, D>> {
        vec![
            self.scale.clone(),
            self.bias.clone(),
            self.mean.clone(),
            self.variance.clone(),
        ]
    }

    fn load_parameters(&mut self, parameters: &[Variable<T, D>]) {
        self.scale = parameters[0].clone();
        self.bias = parameters[1].clone();
        self.mean = parameters[2].clone();
        self.variance = parameters[3].clone();
    }

    fn shape_check(&self, input: &Variable<T, D>) {
        let input_shape = input.get_data().shape();
        let scale_shape = self.scale.get_data().shape();
        let bias_shape = self.bias.get_data().shape();
        let mean_shape = self.mean.get_data().shape();
        let variance_shape = self.variance.get_data().shape();

        assert_eq!(input_shape.len(), 4);
        assert_eq!(scale_shape.len(), 1);
        assert_eq!(bias_shape.len(), 1);
        assert_eq!(mean_shape.len(), 1);
        assert_eq!(variance_shape.len(), 1);
        assert_eq!(scale_shape[0], input_shape[1]);
        assert_eq!(bias_shape[0], input_shape[1]);
        assert_eq!(mean_shape[0], input_shape[1]);
        assert_eq!(variance_shape[0], input_shape[1]);
    }
}

impl<T: Num, D: Device> BatchNorm2d<T, D> {
    pub fn new(input_shape: DimDyn, momentum: f64) -> Self {
        let scale = ones([input_shape[1]]);
        let bias = zeros([input_shape[1]]);
        let mean = zeros([input_shape[1]]);
        let variance = ones([input_shape[1]]);
        let config = BatchNorm2dAutoGradConfig::new(input_shape.slice());
        Self {
            config,
            momentum,
            scale,
            bias,
            mean,
            variance,
        }
    }
}

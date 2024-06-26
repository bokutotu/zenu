use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::{
    creator::{rand::normal, zeros::zeros},
    functions::conv2d::{conv2d, Conv2dConfigs},
    Variable,
};
use zenu_matrix::{device::Device, dim::DimTrait, nn::conv2d::conv2d_out_size, num::Num};

use crate::Layer;

pub struct Conv2d<T: Num, D: Device> {
    filter: Variable<T, D>,
    bias: Option<Variable<T, D>>,
    config: Conv2dConfigs<T>,
    stride: (usize, usize),
    padding: (usize, usize),
}

impl<T: Num, D: Device> Layer<T, D> for Conv2d<T, D> {
    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
        self.shape_check(&input);
        conv2d(
            input,
            self.filter.clone(),
            self.stride,
            self.padding,
            self.bias.clone(),
            Some(self.config.clone()),
        )
    }

    fn parameters(&self) -> Vec<Variable<T, D>> {
        let mut params = vec![self.filter.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn load_parameters(&mut self, parameters: &[Variable<T, D>]) {
        self.filter = parameters[0].clone();
        if parameters.len() > 1 {
            self.bias = Some(parameters[1].clone());
        }
    }

    fn shape_check(&self, input: &Variable<T, D>) {
        let input_shape = input.get_data().shape();
        let filter_shape = self.filter.get_data().shape();
        let bias_shape = self.bias.as_ref().map(|b| b.get_data().shape());

        assert_eq!(input_shape.len(), 4);
        assert_eq!(filter_shape.len(), 4);
        assert_eq!(input_shape[1], filter_shape[1]);
        assert_eq!(input_shape[2], filter_shape[2]);
        assert_eq!(filter_shape[0], bias_shape.map_or(1, |b| b[0]));
    }
}

impl<T: Num, D: Device> Conv2d<T, D> {
    #[must_use]
    pub fn new(
        input_channel: usize,
        output_channel: usize,
        input_image_size: (usize, usize),
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let filter_shape = [output_channel, input_channel, kernel_size.0, kernel_size.1];
        let input_shape = [1, input_channel, input_image_size.0, input_image_size.1];
        let bias = if bias {
            Some(zeros([output_channel]))
        } else {
            None
        };
        let filter = normal(T::zero(), T::one(), None, filter_shape);
        let output_shape = conv2d_out_size(&input_shape, &filter_shape, padding, stride);
        let config = Conv2dConfigs::new(
            input_shape.into(),
            output_shape.into(),
            filter_shape.into(),
            stride,
            padding,
            20,
        );
        Conv2d {
            filter,
            bias,
            config,
            stride,
            padding,
        }
    }
}

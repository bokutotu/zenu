use std::cell::RefCell;

use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use zenu_autograd::{
    creator::{rand::normal, zeros::zeros},
    functions::conv2d::{conv2d, Conv2dConfigs},
    Variable,
};
use zenu_matrix::{device::Device, dim::DimTrait, nn::conv2d::conv2d_out_size, num::Num};

use crate::{Module, StateDict};

#[derive(Serialize, Deserialize)]
#[serde(bound(deserialize = "T: Num + Deserialize<'de>"))]
pub struct Conv2d<T: Num, D: Device> {
    pub filter: Variable<T, D>,
    pub bias: Option<Variable<T, D>>,
    #[serde(skip)]
    config: RefCell<Option<Conv2dConfigs<T>>>,
    stride: (usize, usize),
    padding: (usize, usize),
}

impl<'de, T: Num + Deserialize<'de>, D: Device> StateDict<'de> for Conv2d<T, D> {}

impl<T: Num, D: Device> Module<T, D> for Conv2d<T, D> {
    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
        if self.config.borrow().is_none() {
            let input_shape = input.get_data().shape();
            let filter_shape = self.filter.get_data().shape();
            let output_shape = conv2d_out_size(
                input_shape.slice(),
                filter_shape.slice(),
                self.padding,
                self.stride,
            );
            let config = Conv2dConfigs::new(
                input_shape,
                output_shape.into(),
                filter_shape,
                self.stride,
                self.padding,
                20,
            );
            *self.config.borrow_mut() = Some(config);
        }
        conv2d(
            input,
            self.filter.clone(),
            self.stride,
            self.padding,
            self.bias.clone(),
            Some(self.config.borrow().as_ref().unwrap().clone()),
        )
    }
}

impl<T: Num, D: Device> Conv2d<T, D> {
    #[must_use]
    pub fn new(
        input_channel: usize,
        output_channel: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let filter_shape = [output_channel, input_channel, kernel_size.0, kernel_size.1];
        let bias = if bias {
            let bias = zeros([1, output_channel, 1, 1]);
            bias.set_is_train(true);
            Some(bias)
        } else {
            None
        };
        let filter = normal(T::zero(), T::one(), None, filter_shape);

        filter.set_is_train(true);

        Conv2d {
            filter,
            bias,
            config: RefCell::new(None),
            stride,
            padding,
        }
    }
}

#[cfg(test)]
mod conv2d {
    use zenu_autograd::creator::rand::normal;
    use zenu_matrix::device::cpu::Cpu;

    use crate::{layers::conv2d::Conv2d, Module, StateDict};

    #[test]
    fn conv2d() {
        let input = normal::<f32, _, Cpu>(0.0, 1.0, Some(42), &[2, 3, 5, 5]);
        let conv2d = Conv2d::new(3, 4, (3, 3), (1, 1), (1, 1), true);
        let _output = conv2d.call(input);

        let conv2d_params = conv2d.to_json();

        let conv_2d_json = serde_json::to_string(&conv2d).unwrap();

        let deserialized_conv2d: Conv2d<f32, Cpu> = serde_json::from_str(&conv_2d_json).unwrap();

        let de_parames = deserialized_conv2d.to_json();

        assert_eq!(conv2d_params, de_parames);
    }
}

use std::cell::RefCell;

use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::{
    creator::{rand::normal, zeros::zeros},
    functions::conv2d::{conv2d, Conv2dConfigs},
    Variable,
};
use zenu_matrix::{device::Device, dim::DimTrait, nn::conv2d::conv2d_out_size, num::Num};

use crate::Module;

pub struct Conv2d<T: Num, D: Device> {
    filter: Variable<T, D>,
    bias: Option<Variable<T, D>>,
    config: RefCell<Option<Conv2dConfigs<T>>>,
    stride: (usize, usize),
    padding: (usize, usize),
}

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
            Some(zeros([1, output_channel, 1, 1]))
        } else {
            None
        };
        let filter = normal(T::zero(), T::one(), None, filter_shape);
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
    use zenu_matrix::device::cpu::Cpu;

    use zenu_test::assert_mat_eq_epsilon;

    use crate::{layers::conv2d::Conv2d, Module};

    #[test]
    fn conv2d() {
        let input = zenu_autograd::creator::rand::normal(0.0, 1.0, Some(42), &[2, 3, 5, 5]);
        let conv2d = Conv2d::new(3, 4, (3, 3), (1, 1), (1, 1), true);
        let output = conv2d.call(input);

        let conv2d_params = conv2d.parameters();

        let conv_2d_json = serde_json::to_string(&conv2d).unwrap();

        let deserialized_conv2d: Conv2d<f32, Cpu> = serde_json::from_str(&conv_2d_json).unwrap();

        let de_parames = deserialized_conv2d.parameters();

        assert_eq!(conv2d_params.len(), de_parames.len());
        for (p1, p2) in conv2d_params.iter().zip(de_parames.iter()) {
            assert_mat_eq_epsilon!(p1.get_data(), p2.get_data(), 1e-6);
        }
    }
}

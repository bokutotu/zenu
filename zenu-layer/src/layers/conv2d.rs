use std::collections::HashMap;

use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::{
    creator::{rand::normal, zeros::zeros},
    nn::conv::{conv, ConvConfigs},
    Variable,
};
use zenu_matrix::{
    device::Device,
    dim::{default_stride, DimDyn},
    num::Num,
    shape_stride::ShapeStride,
};

use crate::{Module, Parameters};

pub struct Conv2d<T: Num, D: Device> {
    pub filter: Variable<T, D>,
    pub bias: Option<Variable<T, D>>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilations: (usize, usize),
    config: ConvConfigs<T>,
}

impl<T: Num, D: Device> Module<T, D> for Conv2d<T, D> {
    type Input = Variable<T, D>;
    type Output = Variable<T, D>;

    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
        conv(
            input,
            self.filter.clone(),
            self.bias.clone(),
            self.config.clone(),
        )
    }
}

impl<T: Num, D: Device> Parameters<T, D> for Conv2d<T, D> {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        HashMap::new()
            .into_iter()
            .chain(std::iter::once((
                String::from("conv2d.filter"),
                self.filter.clone(),
            )))
            .collect()
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        self.bias
            .as_ref()
            .map(|bias| {
                HashMap::new()
                    .into_iter()
                    .chain(std::iter::once((String::from("conv2d.bias"), bias.clone())))
                    .collect()
            })
            .unwrap_or_default()
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
        dilations: (usize, usize),
        bias: bool,
    ) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let filter_shape = [output_channel, input_channel, kernel_size.0, kernel_size.1].into();
        let filter_stride = default_stride(filter_shape);
        let filter_shape_stride = ShapeStride::new(filter_shape, filter_stride);
        // input shape葉か変になっているので一回目は適当なshapeにする
        let input_shape: DimDyn = [32, input_channel, 32, 32].into();
        let input_stride = default_stride(input_shape);
        let input_shape_stride = ShapeStride::new(input_shape, input_stride);

        let stride_slice = [stride.0, stride.1];
        let padding_slice = [padding.0, padding.1];
        let dilations_slice = [dilations.0, dilations.1];

        let config = ConvConfigs::new(
            input_shape_stride,
            filter_shape_stride,
            &stride_slice,
            &padding_slice,
            &dilations_slice,
        );
        let bias = if bias {
            let bias = zeros([1, output_channel, 1, 1]);
            bias.set_is_train(true);
            bias.set_name("conv2d.bias");
            Some(bias)
        } else {
            None
        };
        let filter = normal(T::zero(), T::one(), None, filter_shape);

        filter.set_is_train(true);
        filter.set_name("conv2d.filter");

        Conv2d {
            filter,
            bias,
            stride,
            padding,
            dilations,
            config,
        }
    }

    #[must_use]
    pub fn to<Dout: Device>(self) -> Conv2d<T, Dout> {
        let filter = self.filter.to();
        let bias = self.bias.map(|b| b.to());
        Conv2d {
            filter,
            bias,
            config: self.config,
            stride: self.stride,
            padding: self.padding,
            dilations: self.dilations,
        }
    }
}

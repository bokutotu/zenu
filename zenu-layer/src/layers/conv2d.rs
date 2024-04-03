use zenu_autograd::{
    creator::{rand::normal, zeros::zeros},
    functions::conv2d::conv2d,
    Variable,
};
use zenu_matrix::{dim::DimTrait, matrix::MatrixBase, num::Num};

use crate::Layer;

pub struct Conv2d<T: Num> {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    bias: Option<Variable<T>>,
    kernel: Option<Variable<T>>,
}

impl<T: Num> Conv2d<T> {
    #[must_use]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self {
        let bias = if bias {
            Some(zeros([1, out_channels, 1, 1]))
        } else {
            None
        };
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            kernel: None,
            bias,
        }
    }

    #[must_use]
    pub fn kernel(&self) -> Option<Variable<T>> {
        self.kernel.clone()
    }
}

impl<T: Num> Layer<T> for Conv2d<T> {
    fn init_parameters(&mut self, seed: Option<u64>)
    where
        rand_distr::StandardNormal: rand::prelude::Distribution<T>,
    {
        let kernel = normal(
            T::zero(),
            T::one(),
            seed,
            [
                self.out_channels,
                self.in_channels,
                self.kernel_size.0,
                self.kernel_size.1,
            ],
        );
        kernel.set_name("conv2d_kernel");
        self.kernel = Some(kernel);
    }

    fn call(&self, input: Variable<T>) -> Variable<T> {
        self.shape_check(&input);
        conv2d(
            input,
            self.kernel().unwrap(),
            self.bias.clone(),
            self.stride,
            self.padding,
        )
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        if self.bias.is_some() {
            vec![self.kernel().unwrap(), self.bias.clone().unwrap()]
        } else {
            vec![self.kernel().unwrap()]
        }
    }

    fn shape_check(&self, input: &Variable<T>) {
        let input_shape = input.get_data().shape();
        assert_eq!(input_shape.len(), 4, "Input must be 4D tensor");
        assert_eq!(
            input_shape[1], self.in_channels,
            "Input channel must be equal to in_channels"
        );
    }

    fn load_parameters(&mut self, parameters: &[Variable<T>]) {
        self.kernel = Some(parameters[0].clone());
    }
}

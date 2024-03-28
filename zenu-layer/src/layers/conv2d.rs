use zenu_autograd::{creator::rand::normal, functions::conv2d::conv2d, Variable};
use zenu_matrix::{dim::DimTrait, matrix::MatrixBase, num::Num};

use crate::Layer;

pub struct Conv2d<T: Num> {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    kernel: Option<Variable<T>>,
}

impl<T: Num> Conv2d<T> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            kernel: None,
        }
    }

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
        self.kernel = Some(kernel);
    }

    fn call(&self, input: Variable<T>) -> Variable<T> {
        self.shape_check(&input);
        conv2d(input, self.kernel().unwrap(), self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        vec![self.kernel().unwrap()]
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

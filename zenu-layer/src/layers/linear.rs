use crate::Layer;
use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::{
    creator::{rand::normal, zeros::zeros},
    functions::matmul::matmul,
    Variable,
};
use zenu_matrix::{device::Device, num::Num};

pub struct Linear<T: Num, D: Device> {
    weight: Variable<T, D>,
    bias: Option<Variable<T, D>>,
}

impl<T: Num, D: Device> Layer<T, D> for Linear<T, D> {
    fn parameters(&self) -> Vec<Variable<T, D>> {
        if let Some(bias) = &self.bias {
            vec![self.weight.clone(), bias.clone()]
        } else {
            vec![self.weight.clone()]
        }
    }

    fn load_parameters(&mut self, parameters: &[Variable<T, D>]) {
        self.weight = parameters[0].clone();
        if parameters.len() > 1 {
            self.bias = Some(parameters[1].clone());
        } else {
            self.bias = None;
        }
    }

    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
        self.shape_check(&input);
        let output = matmul(input, self.weight.clone());
        if let Some(bias) = &self.bias {
            output + bias.clone()
        } else {
            output
        }
    }

    fn shape_check(&self, input: &Variable<T, D>) {
        // shape check for input and weight
        let input_shape = input.get_data().shape();
        let weight_shape = self.weight.get_data().shape();
        assert_eq!(input_shape[1], weight_shape[0]);

        // shape check for bias
        if let Some(bias) = &self.bias {
            let bias_shape = bias.get_data().shape();
            assert_eq!(bias_shape[0], weight_shape[1]);
        }
    }
}

impl<T: Num, D: Device> Linear<T, D> {
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let weight = normal(T::zero(), T::one(), None, [in_features, out_features]);
        let bias = if use_bias {
            Some(zeros([out_features]))
        } else {
            None
        };
        Self { weight, bias }
    }
}

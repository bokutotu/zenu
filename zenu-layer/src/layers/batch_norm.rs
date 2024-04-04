use zenu_autograd::{
    creator::{ones::ones, zeros::zeros},
    functions::batch_norm::batch_norm,
    Variable,
};
use zenu_matrix::{matrix::MatrixBase, num::Num};

use crate::Layer;

#[derive(Debug)]
pub struct BatchNorm<T: Num> {
    mean: Option<Variable<T>>,
    variance: Option<Variable<T>>,
    decay: Variable<T>,
    epsilon: Variable<T>,
    inv_std: Option<Variable<T>>,
    gamma: Option<Variable<T>>,
    beta: Option<Variable<T>>,
    shape: usize,
}

impl<T: Num> BatchNorm<T> {
    pub fn new(channels: usize, decay: T, epsilon: T) -> Self {
        let decay = Variable::from(decay);
        let epsilon = Variable::from(epsilon);
        BatchNorm {
            mean: None,
            variance: None,
            decay,
            epsilon,
            inv_std: None,
            gamma: None,
            beta: None,
            shape: channels,
        }
    }
}

impl<T: Num> Layer<T> for BatchNorm<T> {
    fn init_parameters(&mut self, _seed: Option<u64>) {
        let d = self.shape;
        let mean = zeros([d]);
        let variance = ones([d]);
        let gamma = ones([d]);
        let beta = zeros([d]);
        self.mean = Some(mean);
        self.variance = Some(variance);
        self.gamma = Some(gamma);
        self.beta = Some(beta);
    }

    fn call(&self, input: Variable<T>) -> Variable<T> {
        self.shape_check(&input);
        let mean = self.mean.clone().unwrap();
        let variance = self.variance.clone().unwrap();
        let decay = self.decay.clone();
        let epsilon = self.epsilon.clone();
        let gamma = self.gamma.clone().unwrap();
        let beta = self.beta.clone().unwrap();
        batch_norm(mean, variance, decay, epsilon, gamma, beta, input)
    }

    fn shape_check(&self, input: &Variable<T>) {
        let channles = input.get_data().shape()[1];
        assert_eq!(
            channles, self.shape,
            "Input shape is not compatible with the layer"
        );
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        let mut parameters = Vec::new();
        if let Some(mean) = &self.mean {
            parameters.push(mean.clone());
        } else {
            panic!("Mean is not initialized");
        }
        if let Some(variance) = &self.variance {
            parameters.push(variance.clone());
        } else {
            panic!("Variance is not initialized");
        }
        parameters.push(self.decay.clone());
        parameters.push(self.epsilon.clone());
        if let Some(inv_std) = &self.inv_std {
            parameters.push(inv_std.clone());
        } else {
            panic!("Inv_std is not initialized");
        }
        if let Some(gamma) = &self.gamma {
            parameters.push(gamma.clone());
        } else {
            panic!("Gamma is not initialized");
        }
        if let Some(beta) = &self.beta {
            parameters.push(beta.clone());
        } else {
            panic!("Beta is not initialized");
        }
        parameters
    }

    fn load_parameters(&mut self, parameters: &[Variable<T>]) {
        let mut parameters = parameters.iter();
        self.mean = Some(parameters.next().unwrap().clone());
        self.variance = Some(parameters.next().unwrap().clone());
        self.decay = parameters.next().unwrap().clone();
        self.epsilon = parameters.next().unwrap().clone();
        self.inv_std = Some(parameters.next().unwrap().clone());
        self.gamma = Some(parameters.next().unwrap().clone());
        self.beta = Some(parameters.next().unwrap().clone());
    }
}

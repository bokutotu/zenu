use rand_distr::{Distribution, StandardNormal};
use zenu_matrix::{device::Device, nn::rnn::RNNWeights as RNNWeightsMat, num::Num};

use crate::{
    creator::{rand::normal, zeros::zeros},
    Variable,
};

#[derive(Clone)]
pub struct RNNWeights<T: Num, D: Device> {
    pub weight_input: Variable<T, D>,
    pub weight_hidden: Variable<T, D>,
    pub bias_input: Variable<T, D>,
    pub bias_hidden: Variable<T, D>,
}

impl<T: Num, D: Device> From<RNNWeightsMat<T, D>> for RNNWeights<T, D> {
    fn from(weights: RNNWeightsMat<T, D>) -> Self {
        Self {
            weight_input: Variable::new(weights.input_weight().new_matrix()),
            weight_hidden: Variable::new(weights.hidden_weight().new_matrix()),
            bias_input: Variable::new(weights.input_bias().new_matrix()),
            bias_hidden: Variable::new(weights.hidden_bias().new_matrix()),
        }
    }
}

impl<T: Num, D: Device> RNNWeights<T, D> {
    #[must_use]
    pub fn init(input_size: usize, hidden_size: usize) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let weight_input = normal(T::zero(), T::one(), None, [hidden_size, input_size]);
        let weight_hidden = normal(T::zero(), T::one(), None, [hidden_size, hidden_size]);
        let bias_input = zeros([hidden_size]);
        let bias_hidden = zeros([hidden_size]);

        RNNWeights {
            weight_input,
            weight_hidden,
            bias_input,
            bias_hidden,
        }
    }
}

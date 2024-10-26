use std::fmt::Debug;

use rand_distr::{Distribution, StandardNormal};
use zenu_matrix::{device::Device, nn::rnn::RNNWeights as RNNWeightsMat, num::Num};

use crate::{
    creator::{rand::normal, zeros::zeros},
    Variable,
};

pub trait CellType: Sized + Clone + Copy + Debug {
    fn hidden_size(hidden_size: usize) -> usize;
}

#[derive(Clone, Copy, Debug)]
pub struct LSTMCell;

impl CellType for LSTMCell {
    fn hidden_size(hidden_size: usize) -> usize {
        hidden_size * 4
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GRUCell;

impl CellType for GRUCell {
    fn hidden_size(hidden_size: usize) -> usize {
        hidden_size * 3
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RNNCell;

impl CellType for RNNCell {
    fn hidden_size(hidden_size: usize) -> usize {
        hidden_size
    }
}

#[derive(Clone)]
pub struct RNNWeights<T: Num, D: Device, C: CellType> {
    pub weight_input: Variable<T, D>,
    pub weight_hidden: Variable<T, D>,
    pub bias_input: Variable<T, D>,
    pub bias_hidden: Variable<T, D>,
    _cell: std::marker::PhantomData<C>,
}

impl<T: Num, D: Device, C: CellType> From<RNNWeightsMat<T, D>> for RNNWeights<T, D, C> {
    fn from(weights: RNNWeightsMat<T, D>) -> Self {
        Self {
            weight_input: Variable::new(weights.input_weight().new_matrix()),
            weight_hidden: Variable::new(weights.hidden_weight().new_matrix()),
            bias_input: Variable::new(weights.input_bias().new_matrix()),
            bias_hidden: Variable::new(weights.hidden_bias().new_matrix()),
            _cell: std::marker::PhantomData,
        }
    }
}

impl<T: Num, D: Device, C: CellType> RNNWeights<T, D, C> {
    #[must_use]
    pub fn new(
        weight_input: Variable<T, D>,
        weight_hidden: Variable<T, D>,
        bias_input: Variable<T, D>,
        bias_hidden: Variable<T, D>,
    ) -> Self {
        Self {
            weight_input,
            weight_hidden,
            bias_input,
            bias_hidden,
            _cell: std::marker::PhantomData,
        }
    }

    #[must_use]
    pub fn init(input_size: usize, hidden_size: usize) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let gate_size = C::hidden_size(hidden_size);
        let weight_input = normal(T::zero(), T::one(), None, [gate_size, input_size]);
        let weight_hidden = normal(T::zero(), T::one(), None, [gate_size, hidden_size]);
        let bias_input = zeros([hidden_size]);
        let bias_hidden = zeros([hidden_size]);

        RNNWeights {
            weight_input,
            weight_hidden,
            bias_input,
            bias_hidden,
            _cell: std::marker::PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct RNNLayerWeights<T: Num, D: Device, C: CellType> {
    pub forward: RNNWeights<T, D, C>,
    pub backward: Option<RNNWeights<T, D, C>>,
}

impl<T: Num, D: Device, C: CellType> RNNLayerWeights<T, D, C> {
    #[must_use]
    pub fn init(input_size: usize, hidden_size: usize, is_bidirectional: bool) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let forward = RNNWeights::init(input_size, hidden_size);
        let backward = if is_bidirectional {
            Some(RNNWeights::init(input_size, hidden_size))
        } else {
            None
        };
        Self { forward, backward }
    }

    #[must_use]
    pub fn new(forward: RNNWeights<T, D, C>, backward: Option<RNNWeights<T, D, C>>) -> Self {
        Self { forward, backward }
    }
}

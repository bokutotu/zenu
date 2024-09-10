use zenu_matrix::{device::Device, num::Num};

use crate::{creator::zeros::zeros, Variable};

use super::RNNOutput;

pub fn rnn_single_layer<T: Num, D: Device>(
    input: Variable<T, D>,
    input_weight: Variable<T, D>,
    hidden_weight: Variable<T, D>,
    input_bias: Variable<T, D>,
    hidden_bias: Variable<T, D>,
    hidden_state: Option<Variable<T, D>>,
) -> RNNOutput<T, D> {
    let seq_len = input.get_shape()[0];
    let batch_size = input.get_shape()[1];
    let hidden_size = hidden_weight.get_shape()[1];

    let hidden_state = hidden_state.unwrap_or(zeros([1, batch_size, hidden_size]));

    todo!();
}

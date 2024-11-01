use zenu_matrix::{device::Device, index::index_dyn_impl::Index, num::Num, slice_dynamic};

use crate::{
    activation::sigmoid::sigmoid,
    concat::concat,
    creator::ones::ones_like,
    functions::{
        index_axis::index_axis, matmul::matmul, slice::slice, stack::stack, tanh::tanh,
        transpose::transpose,
    },
    nn::rnns::weights::{GRUCell, RNNLayerWeights, RNNWeights},
    Variable,
};

fn gru_single_time_step<T: Num, D: Device>(
    x: Variable<T, D>,
    h_prev: Variable<T, D>,
    weight_ih_t: &Variable<T, D>,
    weight_hh_t: &Variable<T, D>,
    bias_ih: &Variable<T, D>,
    bias_hh: &Variable<T, D>,
) -> Variable<T, D> {
    // x: [batch_size, input_size]
    // h_prev: [batch_size, hidden_size]
    // weight_ih_t: [input_size, 3 * hidden_size]
    // weight_hh_t: [hidden_size, 3 * hidden_size]
    // bias_ih: [3 * hidden_size]
    // bias_hh: [3 * hidden_size]

    let gate_x = matmul(x, weight_ih_t.clone()) + bias_ih.clone();
    let gate_h = matmul(h_prev.clone(), weight_hh_t.clone()) + bias_hh.clone();

    let hidden_size = h_prev.get_shape()[1];

    // Split gate_x and gate_h into components
    let gate_x_z = slice(gate_x.clone(), slice_dynamic![.., 0..hidden_size]);
    let gate_x_r = slice(
        gate_x.clone(),
        slice_dynamic![.., hidden_size..2 * hidden_size],
    );
    let gate_x_n = slice(gate_x, slice_dynamic![.., 2 * hidden_size..3 * hidden_size]);

    let gate_h_z = slice(gate_h.clone(), slice_dynamic![.., 0..hidden_size]);
    let gate_h_r = slice(
        gate_h.clone(),
        slice_dynamic![.., hidden_size..2 * hidden_size],
    );
    let gate_h_n = slice(gate_h, slice_dynamic![.., 2 * hidden_size..3 * hidden_size]);

    let z_t = sigmoid(gate_x_z + gate_h_z);
    let r_t = sigmoid(gate_x_r + gate_h_r);
    let n_t = tanh(gate_x_n + r_t * gate_h_n);

    let h_t = (ones_like(&z_t) - z_t.clone()) * h_prev + z_t * n_t;

    h_t
}

fn gru_single_layer_direction<T: Num, D: Device>(
    x: Variable<T, D>,
    mut h: Variable<T, D>,
    weight: &RNNWeights<T, D, GRUCell>,
    reverse: bool,
) -> Vec<Variable<T, D>> {
    let seq_len = x.get_shape()[0];
    let mut out = Vec::new();

    let time_steps: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..seq_len).rev())
    } else {
        Box::new(0..seq_len)
    };

    // Transpose weights once outside the loop
    let weight_ih_t = transpose(weight.weight_input.clone()); // [input_size, 3 * hidden_size]
    let weight_hh_t = transpose(weight.weight_hidden.clone()); // [hidden_size, 3 * hidden_size]
    let bias_ih = &weight.bias_input;
    let bias_hh = &weight.bias_hidden;

    for time_step in time_steps {
        let x_t = index_axis(x.clone(), Index::new(0, time_step));
        h = gru_single_time_step(x_t, h.clone(), &weight_ih_t, &weight_hh_t, bias_ih, bias_hh);
        out.push(h.clone());
    }

    out
}

#[expect(clippy::too_many_arguments)]
#[must_use]
fn gru_single_layer<T: Num, D: Device>(
    x: Variable<T, D>,
    h_forward: Variable<T, D>,
    h_backward: Option<Variable<T, D>>,
    weight_forward: &RNNWeights<T, D, GRUCell>,
    weight_backward: Option<&RNNWeights<T, D, GRUCell>>,
    bidirectional: bool,
) -> Variable<T, D> {
    let forward_output = gru_single_layer_direction(x.clone(), h_forward, weight_forward, false);

    if bidirectional {
        let h_backward =
            h_backward.expect("Hidden state for backward pass is required in bidirectional mode");
        let weight_backward =
            weight_backward.expect("Weights for backward pass are required in bidirectional mode");

        let backward_output = gru_single_layer_direction(x, h_backward, weight_backward, true);

        // Reverse the backward output to match the forward direction
        let backward_output_rev: Vec<_> = backward_output.into_iter().rev().collect();

        // Concatenate outputs from forward and backward passes
        let seq_len = forward_output.len();
        let mut outputs = Vec::new();
        for t in 0..seq_len {
            let f = forward_output[t].clone();
            let b = backward_output_rev[t].clone();
            let output_t = stack(&[f, b], 1); // Concatenate along the hidden_size dimension
            outputs.push(output_t);
        }

        concat(&outputs)
    } else {
        concat(&forward_output)
    }
}

/// h shape is [`num_layers` * `num_directions`, `batch_size`, `hidden_size`]
#[expect(clippy::needless_pass_by_value)]
#[must_use]
pub fn gru<T: Num, D: Device>(
    x: Variable<T, D>,
    h: Variable<T, D>,
    weights: &[RNNLayerWeights<T, D, GRUCell>],
    bidirectional: bool,
) -> Variable<T, D> {
    let mut state = x;
    let num_directions = if bidirectional { 2 } else { 1 };

    for (layer, layer_weight) in weights.iter().enumerate() {
        let h_forward = index_axis(h.clone(), Index::new(0, layer * num_directions));
        let h_backward = if bidirectional {
            Some(index_axis(
                h.clone(),
                Index::new(0, layer * num_directions + 1),
            ))
        } else {
            None
        };

        let output = gru_single_layer(
            state.clone(),
            h_forward,
            h_backward,
            &layer_weight.forward,
            layer_weight.backward.as_ref(),
            bidirectional,
        );

        state = output;
    }

    state
}

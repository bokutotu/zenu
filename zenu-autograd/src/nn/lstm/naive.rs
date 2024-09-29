use rand_distr::{Distribution, StandardNormal};
use zenu_matrix::{device::Device, index::index_dyn_impl::Index, num::Num};

use crate::{
    concat::concat,
    creator::{rand::normal, zeros::zeros},
    functions::{
        activation::sigmoid::sigmoid, index_axis::index_axis, matmul::matmul, stack::stack,
        tanh::tanh, transpose::transpose,
    },
    Variable,
};

#[expect(clippy::needless_pass_by_value, clippy::similar_names)]
fn lstm_single_time_step<T: Num, D: Device>(
    x: Variable<T, D>,
    h_prev: Variable<T, D>,
    c_prev: Variable<T, D>,
    weight_ih: Variable<T, D>,
    weight_hh: Variable<T, D>,
    bias_ih: Variable<T, D>,
    bias_hh: Variable<T, D>,
) -> (Variable<T, D>, Variable<T, D>) {
    // x: [batch_size, input_size]
    // h_prev: [batch_size, hidden_size]
    // c_prev: [batch_size, hidden_size]
    // weight_ih: [4 * hidden_size, input_size]
    // weight_hh: [4 * hidden_size, hidden_size]
    // bias_ih: [4 * hidden_size]
    // bias_hh: [4 * hidden_size]

    let weight_ih_t = transpose(weight_ih);
    let weight_hh_t = transpose(weight_hh);

    let gates = matmul(x, weight_ih_t) + matmul(h_prev, weight_hh_t) + bias_ih + bias_hh;
    // gates: [batch_size, 4 * hidden_size]

    let hidden_size = h_prev.get_shape()[1];

    let i_t = sigmoid(index_axis(gates.clone(), Index::new(1, 0..hidden_size)));
    let f_t = sigmoid(index_axis(
        gates.clone(),
        Index::new(1, hidden_size..2 * hidden_size),
    ));
    let g_t = tanh(index_axis(
        gates.clone(),
        Index::new(1, 2 * hidden_size..3 * hidden_size),
    ));
    let o_t = sigmoid(index_axis(
        gates,
        Index::new(1, 3 * hidden_size..4 * hidden_size),
    ));

    let c_t = f_t * c_prev + i_t * g_t;
    let h_t = o_t * tanh(c_t.clone());

    (h_t, c_t)
}

#[expect(clippy::needless_pass_by_value)]
#[must_use]
fn lstm_single_layer<T: Num, D: Device>(
    x: Variable<T, D>,
    h_forward: Variable<T, D>,
    c_forward: Variable<T, D>,
    h_backward: Option<Variable<T, D>>,
    c_backward: Option<Variable<T, D>>,
    weight_forward: &LSTMWeights<T, D>,
    weight_backward: Option<&LSTMWeights<T, D>>,
    bidirectional: bool,
) -> Variable<T, D> {
    let forward_output =
        lstm_single_layer_direction(x.clone(), h_forward, c_forward, weight_forward, false);

    if bidirectional {
        let h_backward =
            h_backward.expect("Hidden state for backward pass is required in bidirectional mode");
        let c_backward =
            c_backward.expect("Cell state for backward pass is required in bidirectional mode");
        let weight_backward =
            weight_backward.expect("Weights for backward pass are required in bidirectional mode");

        let backward_output =
            lstm_single_layer_direction(x, h_backward, c_backward, weight_backward, true);

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

fn lstm_single_layer_direction<T: Num, D: Device>(
    x: Variable<T, D>,
    mut h: Variable<T, D>,
    mut c: Variable<T, D>,
    weight: &LSTMWeights<T, D>,
    reverse: bool,
) -> Vec<Variable<T, D>> {
    let seq_len = x.get_shape()[0];
    let mut out = Vec::new();

    let time_steps: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..seq_len).rev())
    } else {
        Box::new(0..seq_len)
    };

    let weight_ih = weight.weight_ih.clone();
    let weight_hh = weight.weight_hh.clone();
    let bias_ih = weight.bias_ih.clone();
    let bias_hh = weight.bias_hh.clone();

    for time_step in time_steps {
        let x_t = index_axis(x.clone(), Index::new(0, time_step));
        let (h_t, c_t) = lstm_single_time_step(
            x_t,
            h.clone(),
            c.clone(),
            weight_ih.clone(),
            weight_hh.clone(),
            bias_ih.clone(),
            bias_hh.clone(),
        );
        out.push(h_t.clone());
        h = h_t;
        c = c_t;
    }

    out
}

#[derive(Clone)]
pub struct LSTMWeights<T: Num, D: Device> {
    pub weight_ih: Variable<T, D>, // [4 * hidden_size, input_size]
    pub weight_hh: Variable<T, D>, // [4 * hidden_size, hidden_size]
    pub bias_ih: Variable<T, D>,   // [4 * hidden_size]
    pub bias_hh: Variable<T, D>,   // [4 * hidden_size]
}

impl<T: Num, D: Device> LSTMWeights<T, D> {
    #[must_use]
    pub fn init(input_size: usize, hidden_size: usize) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let gate_size = 4 * hidden_size;
        let weight_ih = normal(T::zero(), T::one(), None, [gate_size, input_size]);
        let weight_hh = normal(T::zero(), T::one(), None, [gate_size, hidden_size]);
        let bias_ih = zeros([gate_size]);
        let bias_hh = zeros([gate_size]);

        LSTMWeights {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }
}

#[derive(Clone)]
pub struct LSTMLayerWeights<T: Num, D: Device> {
    pub forward: LSTMWeights<T, D>,
    pub backward: Option<LSTMWeights<T, D>>,
}

impl<T: Num, D: Device> LSTMLayerWeights<T, D> {
    #[must_use]
    pub fn init(input_size: usize, hidden_size: usize, is_bidirectional: bool) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let forward = LSTMWeights::init(input_size, hidden_size);
        let backward = if is_bidirectional {
            Some(LSTMWeights::init(input_size, hidden_size))
        } else {
            None
        };
        Self { forward, backward }
    }

    #[must_use]
    pub fn new(forward: LSTMWeights<T, D>, backward: Option<LSTMWeights<T, D>>) -> Self {
        Self { forward, backward }
    }
}

/// h and c shapes are [`num_layers` * `num_directions`, `batch_size`, `hidden_size`]
#[expect(clippy::needless_pass_by_value)]
#[must_use]
pub fn lstm<T: Num, D: Device>(
    x: Variable<T, D>,
    h: Variable<T, D>,
    c: Variable<T, D>,
    weights: &[LSTMLayerWeights<T, D>],
    bidirectional: bool,
) -> Variable<T, D> {
    let mut state = x;
    let num_directions = if bidirectional { 2 } else { 1 };

    for (layer, layer_weight) in weights.iter().enumerate() {
        let h_forward = index_axis(h.clone(), Index::new(0, layer * num_directions));
        let c_forward = index_axis(c.clone(), Index::new(0, layer * num_directions));
        let (h_backward, c_backward) = if bidirectional {
            (
                Some(index_axis(
                    h.clone(),
                    Index::new(0, layer * num_directions + 1),
                )),
                Some(index_axis(
                    c.clone(),
                    Index::new(0, layer * num_directions + 1),
                )),
            )
        } else {
            (None, None)
        };

        let output = lstm_single_layer(
            state.clone(),
            h_forward,
            c_forward,
            h_backward,
            c_backward,
            &layer_weight.forward,
            layer_weight.backward.as_ref(),
            bidirectional,
        );
        println!("output shape: {:?}", output.get_shape());

        // For the next layer, state is the output from this layer
        state = output;
    }

    state
}

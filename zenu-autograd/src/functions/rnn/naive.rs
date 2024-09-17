use zenu_matrix::{device::Device, index::index_dyn_impl::Index, num::Num};

use crate::{
    concat::concat,
    functions::{
        activation::relu::relu, index_axis::index_axis, matmul::matmul, stack::stack, tanh::tanh,
        transpose::transpose,
    },
    Variable,
};

fn rnn_single_time_step<T: Num, D: Device, F>(
    x: Variable<T, D>,
    h: Variable<T, D>,
    weight_input: Variable<T, D>,
    weight_hidden: Variable<T, D>,
    bias_input: Variable<T, D>,
    bias_hidden: Variable<T, D>,
    activation: F,
) -> Variable<T, D>
where
    F: Fn(Variable<T, D>) -> Variable<T, D>,
{
    let input_state = matmul(x, weight_input);
    let input_state = input_state + bias_input;

    let hidden_state = matmul(h, weight_hidden);
    let hidden_state = hidden_state + bias_hidden;

    let state = input_state + hidden_state;
    activation(state)
}

#[expect(clippy::needless_pass_by_value)]
#[must_use]
fn rnn_single_layer<T: Num, D: Device, F>(
    x: Variable<T, D>,
    h_forward: Variable<T, D>,
    h_backward: Option<Variable<T, D>>,
    weight_forward: &RNNWeights<T, D>,
    weight_backward: Option<&RNNWeights<T, D>>,
    activation: F,
    bidirectional: bool,
) -> Variable<T, D>
where
    F: Fn(Variable<T, D>) -> Variable<T, D> + Copy,
{
    let forward_output =
        rnn_single_layer_direction(x.clone(), h_forward, weight_forward, activation, false);

    if bidirectional {
        let h_backward =
            h_backward.expect("Hidden state for backward pass is required in bidirectional mode");
        let weight_backward =
            weight_backward.expect("Weights for backward pass are required in bidirectional mode");
        let backward_output =
            rnn_single_layer_direction(x.clone(), h_backward, weight_backward, activation, true);

        // Reverse the backward output to match the forward direction
        let backward_output_rev: Vec<_> = backward_output.into_iter().rev().collect();

        // Concatenate outputs from forward and backward passes
        let seq_len = forward_output.len();
        let mut outputs = Vec::new();
        for t in 0..seq_len {
            let f = forward_output[t].clone();
            let b = backward_output_rev[t].clone();
            let output_t = concat(&[f, b]); // Concatenate along the hidden_size dimension
            outputs.push(output_t);
        }

        stack(&outputs, 0)
    } else {
        concat(&forward_output)
    }
}

#[expect(clippy::needless_pass_by_value)]
fn rnn_single_layer_direction<T: Num, D: Device, F>(
    x: Variable<T, D>,
    mut h: Variable<T, D>,
    weight: &RNNWeights<T, D>,
    activation: F,
    reverse: bool,
) -> Vec<Variable<T, D>>
where
    F: Fn(Variable<T, D>) -> Variable<T, D> + Copy,
{
    let seq_len = x.get_shape()[0];
    let mut out = Vec::new();

    let time_steps: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..seq_len).rev())
    } else {
        Box::new(0..seq_len)
    };

    let input_weight = weight.weight_input.clone();
    let hidden_weight = weight.weight_hidden.clone();
    let bias_input = weight.bias_input.clone();
    let bias_hidden = weight.bias_hidden.clone();

    let input_weight = transpose(input_weight);
    let hidden_weight = transpose(hidden_weight);

    for time_step in time_steps {
        let x_t = index_axis(x.clone(), Index::new(0, time_step));
        let h_t = h.clone();
        let out_t = rnn_single_time_step(
            x_t,
            h_t,
            input_weight.clone(),
            hidden_weight.clone(),
            bias_input.clone(),
            bias_hidden.clone(),
            activation,
        );
        out.push(out_t.clone());
        h = out_t;
    }

    out
}

#[derive(Clone)]
pub struct RNNWeights<T: Num, D: Device> {
    pub weight_input: Variable<T, D>,
    pub weight_hidden: Variable<T, D>,
    pub bias_input: Variable<T, D>,
    pub bias_hidden: Variable<T, D>,
}

#[derive(Clone)]
pub struct RNNLayerWeights<T: Num, D: Device> {
    pub forward: RNNWeights<T, D>,
    pub backward: Option<RNNWeights<T, D>>,
}

/// h shape is [`num_layers` * `num_directions`, `batch_size`, `hidden_size`]
#[expect(clippy::needless_pass_by_value)]
#[must_use]
fn rnn<T: Num, D: Device, F>(
    x: Variable<T, D>,
    h: Variable<T, D>,
    weights: &[RNNLayerWeights<T, D>],
    activation: F,
    bidirectional: bool,
) -> Variable<T, D>
where
    F: Fn(Variable<T, D>) -> Variable<T, D> + Copy,
{
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

        let output = rnn_single_layer(
            state.clone(),
            h_forward,
            h_backward,
            &layer_weight.forward,
            layer_weight.backward.as_ref(),
            activation,
            bidirectional,
        );

        // For the next layer, state is the output from this layer
        state = output;
    }

    state
}

#[must_use]
pub fn rnn_tanh<T: Num, D: Device>(
    x: Variable<T, D>,
    h: Variable<T, D>,
    weights: &[RNNLayerWeights<T, D>],
    bidirectional: bool,
) -> Variable<T, D> {
    rnn(x, h, weights, tanh, bidirectional)
}

#[must_use]
pub fn rnn_relu<T: Num, D: Device>(
    x: Variable<T, D>,
    h: Variable<T, D>,
    weights: &[RNNLayerWeights<T, D>],
    bidirectional: bool,
) -> Variable<T, D> {
    h.set_name("h");
    rnn(x, h, weights, relu, bidirectional)
}

#[cfg(test)]
mod rnn_test {
    use std::collections::HashMap;

    use zenu_matrix::{
        device::{cpu::Cpu, Device},
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, read_test_case_from_json_val, run_test};

    use crate::{
        creator::{ones::ones, zeros::zeros},
        functions::rnn::naive::{rnn_relu, RNNLayerWeights},
        Variable,
    };

    use super::RNNWeights;

    fn load_rnn_weight_from_json<D: Device>(
        path: &str,
        idx: usize,
        bidirectional: bool,
    ) -> RNNLayerWeights<f32, D> {
        let mats: HashMap<String, Matrix<Owned<f32>, DimDyn, Cpu>> =
            read_test_case_from_json_val!(path);

        let input_weight = mats
            .get(&format!("rnn.weight_ih_l{}", idx))
            .unwrap()
            .clone();
        let hidden_weight = mats
            .get(&format!("rnn.weight_hh_l{}", idx))
            .unwrap()
            .clone();
        let input_bias = mats.get(&format!("rnn.bias_ih_l{}", idx)).unwrap().clone();
        let hidden_bias = mats.get(&format!("rnn.bias_hh_l{}", idx)).unwrap().clone();

        let forward = RNNWeights {
            weight_input: Variable::<f32, D>::new(input_weight.to::<D>()),
            weight_hidden: Variable::<f32, D>::new(hidden_weight.to::<D>()),
            bias_input: Variable::<f32, D>::new(input_bias.to::<D>()),
            bias_hidden: Variable::<f32, D>::new(hidden_bias.to::<D>()),
        };

        let reverse = if bidirectional {
            let input_weight_rev = mats
                .get(&format!("rnn.weight_ih_l{}_reverse", idx))
                .unwrap()
                .clone();
            let hidden_weight_rev = mats
                .get(&format!("rnn.weight_hh_l{}_reverse", idx))
                .unwrap()
                .clone();
            let input_bias_rev = mats
                .get(&format!("rnn.bias_ih_l{}_reverse", idx))
                .unwrap()
                .clone();
            let hidden_bias_rev = mats
                .get(&format!("rnn.bias_hh_l{}_reverse", idx))
                .unwrap()
                .clone();
            Some(RNNWeights {
                weight_input: Variable::<f32, D>::new(input_weight_rev.to::<D>()),
                weight_hidden: Variable::<f32, D>::new(hidden_weight_rev.to::<D>()),
                bias_input: Variable::<f32, D>::new(input_bias_rev.to::<D>()),
                bias_hidden: Variable::<f32, D>::new(hidden_bias_rev.to::<D>()),
            })
        } else {
            None
        };

        RNNLayerWeights {
            forward,
            backward: reverse,
        }
    }

    fn rnn_test_single_layer<D: Device>(path: &str) {
        let mats: HashMap<String, Matrix<Owned<f32>, DimDyn, Cpu>> =
            read_test_case_from_json_val!(path);

        let input = mats.get("input").unwrap().clone();
        let input = Variable::<f32, D>::new(input.to::<D>());
        let weight = load_rnn_weight_from_json::<D>(path, 0, false);

        let batch_size = input.get_shape()[1];
        let hidden_size = weight.forward.weight_hidden.get_shape()[0];

        let output = rnn_relu(
            input.clone(),
            zeros([1, batch_size, hidden_size]),
            &[weight.clone()],
            false,
        );
        output.backward();
        let expected = mats.get("output").unwrap().clone();
        assert_val_eq!(output, expected.to::<D>(), 1e-5);
        assert_val_eq_grad!(
            input,
            mats.get("input_grad").unwrap().clone().to::<D>(),
            1e-5
        );

        assert_val_eq_grad!(
            weight.forward.weight_input,
            mats.get("rnn.weight_ih_l0_grad").unwrap().clone().to::<D>(),
            1e-5
        );
    }

    fn rnn_single_layer<D: Device>() {
        rnn_test_single_layer::<D>("../test_data_json/rnn_fwd_bkwd_single.json");
    }
    run_test!(rnn_single_layer, rnn_single_layer_cpu, rnn_single_layer_gpu);

    fn rnn_single_layer_seq_1<D: Device>() {
        rnn_test_single_layer::<D>("../test_data_json/rnn_fwd_bkwd_single_seq_len_1.json");
    }
    run_test!(
        rnn_single_layer_seq_1,
        rnn_single_layer_seq_1_cpu,
        rnn_single_layer_seq_1_gpu
    );

    fn rnn_very_small_test<D: Device>() {
        let input = ones::<f32, _, D>([3, 1, 1]);
        let input_weight =
            Variable::from(Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![0.5], [1, 1]));
        let hidden_weight =
            Variable::from(Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![2.], [1, 1]));
        let input_bias = zeros([1]);
        let hidden_bias = zeros([1]);

        let weight_forward = RNNWeights {
            weight_input: input_weight,
            weight_hidden: hidden_weight,
            bias_input: input_bias,
            bias_hidden: hidden_bias,
        };

        let weights = vec![RNNLayerWeights {
            forward: weight_forward,
            backward: None,
        }];

        let output = rnn_relu(input.clone(), zeros([1, 1, 1]), &weights, false);
        output.backward();

        let expected = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![0.5, 1.5, 3.5], [3, 1, 1]);
        assert_val_eq!(output.clone(), expected, 1e-5);
        let input_grad = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![3.5, 1.5, 0.5], [3, 1, 1]);
        assert_val_eq_grad!(input, input_grad, 1e-5);
    }
    run_test!(
        rnn_very_small_test,
        rnn_very_small_test_cpu,
        rnn_very_small_test_gpu
    );
}

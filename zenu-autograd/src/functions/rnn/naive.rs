use zenu_matrix::{device::Device, index::index_dyn_impl::Index, num::Num};

use crate::{
    concat::concat,
    functions::{
        activation::relu::relu, index_axis::index_axis, matmul::matmul, tanh::tanh,
        transpose::transpose,
    },
    Variable,
};

pub fn rnn_single_time_step<T: Num, D: Device, F>(
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
pub fn rnn_single_layer<T: Num, D: Device, F>(
    x: Variable<T, D>,
    mut h: Variable<T, D>,
    weight_input: Variable<T, D>,
    weight_hidden: Variable<T, D>,
    bias_input: Variable<T, D>,
    bias_hidden: Variable<T, D>,
    activation: F,
) -> Variable<T, D>
where
    F: Fn(Variable<T, D>) -> Variable<T, D> + Copy,
{
    let seq_len = x.get_shape()[0];
    let mut out = Vec::new();

    x.set_name("x");

    for time_step in 0..seq_len {
        let x_t = index_axis(x.clone(), Index::new(0, time_step));
        let h_t = h.clone();
        let out_t = rnn_single_time_step(
            x_t,
            h_t,
            weight_input.clone(),
            weight_hidden.clone(),
            bias_input.clone(),
            bias_hidden.clone(),
            activation,
        );
        out_t.set_name(&format!("output_{time_step}"));
        out.push(out_t.clone());
        h = out_t;
    }

    concat(&out)
}

pub struct RNNWeights<T: Num, D: Device> {
    pub weight_input: Variable<T, D>,
    pub weight_hidden: Variable<T, D>,
    pub bias_input: Variable<T, D>,
    pub bias_hidden: Variable<T, D>,
}

/// h shape is [`num_layers`, `batch_size`, `hidden_size`]
#[expect(clippy::needless_pass_by_value)]
#[must_use]
pub fn rnn<T: Num, D: Device, F>(
    mut x: Variable<T, D>,
    h: Variable<T, D>,
    weights: &[RNNWeights<T, D>],
    activation: F,
) -> Variable<T, D>
where
    F: Fn(Variable<T, D>) -> Variable<T, D> + Copy,
{
    for (idx, weight) in weights.iter().enumerate() {
        let h_sliced = index_axis(h.clone(), Index::new(0, idx));
        x = rnn_single_layer(
            x,
            h_sliced,
            transpose(weight.weight_input.clone()),
            transpose(weight.weight_hidden.clone()),
            weight.bias_input.clone(),
            weight.bias_hidden.clone(),
            activation,
        );
    }

    x
}

#[must_use]
pub fn rnn_tanh<T: Num, D: Device>(
    x: Variable<T, D>,
    h: Variable<T, D>,
    weights: &[RNNWeights<T, D>],
) -> Variable<T, D> {
    rnn(x, h, weights, tanh)
}

#[must_use]
pub fn rnn_relu<T: Num, D: Device>(
    x: Variable<T, D>,
    h: Variable<T, D>,
    weights: &[RNNWeights<T, D>],
) -> Variable<T, D> {
    h.set_name("h");
    rnn(x, h, weights, relu)
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
        functions::rnn::naive::rnn_relu,
        Variable,
    };

    use super::RNNWeights;

    fn rnn_test_single_layer<D: Device>(path: &str) {
        let mats: HashMap<String, Matrix<Owned<f32>, DimDyn, Cpu>> =
            read_test_case_from_json_val!(path);

        let input = mats.get("input").unwrap().clone();
        let input_weight = mats.get("rnn.weight_ih_l0").unwrap().clone();
        let hidden_weight = mats.get("rnn.weight_hh_l0").unwrap().clone();
        let input_bias = mats.get("rnn.bias_ih_l0").unwrap().clone();
        let hidden_bias = mats.get("rnn.bias_hh_l0").unwrap().clone();

        let input = Variable::<f32, D>::new(input.to::<D>());
        let input_weight = Variable::<f32, D>::new(input_weight.to::<D>());
        let hidden_weight = Variable::<f32, D>::new(hidden_weight.to::<D>());
        let input_bias = Variable::<f32, D>::new(input_bias.to::<D>());
        let hidden_bias = Variable::<f32, D>::new(hidden_bias.to::<D>());

        let weight = RNNWeights {
            weight_input: input_weight,
            weight_hidden: hidden_weight.clone(),
            bias_input: input_bias,
            bias_hidden: hidden_bias,
        };

        let weight = vec![weight];

        let batch_size = input.get_shape()[1];
        let hidden_size = hidden_weight.get_shape()[0];

        let output = rnn_relu(input.clone(), zeros([1, batch_size, hidden_size]), &weight);
        output.backward();
        let expected = mats.get("output").unwrap().clone();
        assert_val_eq!(output, expected.to::<D>(), 1e-5);
        assert_val_eq_grad!(
            input,
            mats.get("input_grad").unwrap().clone().to::<D>(),
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

        let output = rnn_relu(
            input.clone(),
            zeros([1, 1, 1]),
            &[RNNWeights {
                weight_input: input_weight,
                weight_hidden: hidden_weight,
                bias_input: input_bias,
                bias_hidden: hidden_bias,
            }],
        );
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

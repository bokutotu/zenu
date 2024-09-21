use std::collections::HashMap;

use zenu_test::assert_mat_eq_epsilon;
use zenu_test::read_test_case_from_json_val;

use zenu_matrix::device::cpu::Cpu;
use zenu_matrix::matrix::Owned;
use zenu_matrix::{device::nvidia::Nvidia, dim::DimDyn, matrix::Matrix, nn::rnn::*};

fn get_rnn_weights_from_json(
    matrix_map: &std::collections::HashMap<String, Matrix<Owned<f32>, DimDyn, Cpu>>,
    num_layers: usize,
    bidirectional: bool,
    suffix: &str,
) -> Vec<RNNWeightsMat<f32, Cpu>> {
    let mut weights = Vec::new();
    for layer_id in 0..num_layers {
        let input_weight: Matrix<Owned<f32>, DimDyn, Cpu> = matrix_map
            .get(&format!("rnn.weight_ih_l{layer_id}{suffix}"))
            .unwrap()
            .clone();
        let hidden_weight = matrix_map
            .get(&format!("rnn.weight_hh_l{layer_id}{suffix}"))
            .unwrap()
            .clone();
        let input_bias = matrix_map
            .get(&format!("rnn.bias_ih_l{layer_id}{suffix}"))
            .unwrap()
            .clone();
        let hidden_bias = matrix_map
            .get(&format!("rnn.bias_hh_l{layer_id}{suffix}"))
            .unwrap()
            .clone();

        let rnn_weights = RNNWeightsMat::new(input_weight, hidden_weight, input_bias, hidden_bias);
        weights.push(rnn_weights);

        if bidirectional {
            let input_weight: Matrix<Owned<f32>, DimDyn, Cpu> = matrix_map
                .get(&format!("rnn.weight_ih_l{layer_id}_reverse{suffix}"))
                .unwrap()
                .clone();
            let hidden_weight = matrix_map
                .get(&format!("rnn.weight_hh_l{layer_id}_reverse{suffix}"))
                .unwrap()
                .clone();
            let input_bias = matrix_map
                .get(&format!("rnn.bias_ih_l{layer_id}_reverse{suffix}"))
                .unwrap()
                .clone();
            let hidden_bias = matrix_map
                .get(&format!("rnn.bias_hh_l{layer_id}_reverse{suffix}"))
                .unwrap()
                .clone();

            let rnn_weights =
                RNNWeightsMat::new(input_weight, hidden_weight, input_bias, hidden_bias);
            weights.push(rnn_weights);
        }
    }
    weights
}

fn assert_grad(expected: &[RNNWeightsMat<f32, Cpu>], actual: &[RNNWeightsMat<f32, Cpu>]) {
    for (expected, actual) in expected.iter().zip(actual.iter()) {
        assert_mat_eq_epsilon!(expected.input_weight(), actual.input_weight(), 1e-5);
        assert_mat_eq_epsilon!(expected.hidden_weight(), actual.hidden_weight(), 1e-5);
        assert_mat_eq_epsilon!(expected.input_bias(), actual.input_bias(), 1e-5);
        assert_mat_eq_epsilon!(expected.hidden_bias(), actual.hidden_bias(), 1e-5);
    }
}

fn before_run(
    map: &HashMap<String, Matrix<Owned<f32>, DimDyn, Cpu>>,
    bidirectional: bool,
) -> (usize, usize, usize) {
    let input = map.get("input").unwrap().clone();
    let output = map.get("output").unwrap().clone();
    let input_size = input.shape()[2];
    let hidden_size = output.shape()[2] / if bidirectional { 2 } else { 1 };
    let batch_size = input.shape()[1];
    (input_size, hidden_size, batch_size)
}

fn init_weights(
    desc: &RNNDescriptor<f32>,
    map: &HashMap<String, Matrix<Owned<f32>, DimDyn, Cpu>>,
    bidirectional: bool,
    num_layers: usize,
) -> Matrix<Owned<f32>, DimDyn, Nvidia> {
    let weight_num_elm = desc.get_weight_num_elems();
    let mut w = Matrix::<Owned<f32>, DimDyn, Nvidia>::alloc([weight_num_elm]);
    desc.load_rnn_weights(
        w.to_ref_mut().as_mut_ptr().cast(),
        get_rnn_weights_from_json(map, num_layers, bidirectional, ""),
    )
    .unwrap();
    w
}

fn rnn(json_path: String, num_layers: usize, bidirectional: bool) {
    let matrix_map = read_test_case_from_json_val!(json_path);
    let (input_size, hidden_size, batch_size) = before_run(&matrix_map, bidirectional);
    let mut desc = RNNDescriptor::<f32>::new_rnn_relu(
        bidirectional,
        0.0,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
    );

    let weight = init_weights(&desc, &matrix_map, bidirectional, num_layers);

    let input = matrix_map.get("input").unwrap().clone().to::<Nvidia>();
    let y = desc.rnn_fwd(input.to_ref(), None, weight.to_ref(), true);
    let output = matrix_map.get("output").unwrap().clone().to::<Nvidia>();
    assert_mat_eq_epsilon!(y.y.to_ref(), output, 1e-5);

    let dy = Matrix::ones_like(&y.y);

    let dx = desc.rnn_bkwd_data(
        input.shape(),
        y.y.to_ref(),
        dy.to_ref(),
        None,
        None,
        weight.to_ref(),
    );

    assert_mat_eq_epsilon!(
        dx.dx.to_ref(),
        matrix_map.get("input_grad").unwrap().clone().to::<Nvidia>(),
        1e-5
    );
    let mut dw = desc.rnn_bkwd_weights(input.to_ref(), None, y.y.to_ref());

    let params = desc.store_rnn_weights::<Cpu>(dw.to_ref_mut().as_ptr() as *mut u8);

    let weights_grad = get_rnn_weights_from_json(&matrix_map, num_layers, bidirectional, "_grad");
    assert_grad(&weights_grad, &params);
}

fn lstm(json_path: String, num_layers: usize, bidirectional: bool) {
    let matrix_map = read_test_case_from_json_val!(json_path);
    let (input_size, hidden_size, batch_size) = before_run(&matrix_map, bidirectional);
    let mut desc = RNNDescriptor::<f32>::lstm(
        bidirectional,
        0.0,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
    );

    let weight = init_weights(&desc, &matrix_map, bidirectional, num_layers);

    let input = matrix_map.get("input").unwrap().clone().to::<Nvidia>();
    let y = desc.lstm_fwd(input.to_ref(), None, None, weight.to_ref(), true);
    let output = matrix_map.get("output").unwrap().clone().to::<Nvidia>();
    assert_mat_eq_epsilon!(y.y.to_ref(), output, 1e-5);

    let dy = Matrix::ones_like(&y.y);

    let dx = desc.lstm_bkwd_data(
        input.shape(),
        y.y.to_ref(),
        dy.to_ref(),
        None,
        None,
        None,
        None,
        weight.to_ref(),
    );

    assert_mat_eq_epsilon!(
        dx.dx.to_ref(),
        matrix_map.get("input_grad").unwrap().clone().to::<Nvidia>(),
        1e-5
    );
    let mut dw = desc.lstm_bkwd_weights(input.to_ref(), None, None, y.y.to_ref());

    let params = desc.store_rnn_weights::<Cpu>(dw.to_ref_mut().as_ptr() as *mut u8);

    let weights_grad = get_rnn_weights_from_json(&matrix_map, num_layers, bidirectional, "_grad");
    assert_grad(&weights_grad, &params);
}

#[test]
fn test_rnn_seq_len_1() {
    rnn(
        "../test_data_json/rnn_fwd_bkwd_single_seq_len_1.json".to_string(),
        1,
        false,
    );
}

#[test]
fn test_rnn_seq_len_3() {
    rnn(
        "../test_data_json/rnn_fwd_bkwd_single_seq_len_3.json".to_string(),
        1,
        false,
    );
}

#[test]
fn test_rnn_seq_len_5() {
    rnn(
        "../test_data_json/rnn_fwd_bkwd_single.json".to_string(),
        1,
        false,
    );
}

#[test]
fn test_rnn_seq_len_5_num_layer_2_bidirectional() {
    rnn(
        "../test_data_json/rnn_fwd_bkwd_bidirectional_2_layers.json".to_string(),
        2,
        true,
    );
}

#[test]
fn lstm_small() {
    lstm(
        "../test_data_json/lstm_fwd_bkwd_small.json".to_string(),
        1,
        false,
    );
}

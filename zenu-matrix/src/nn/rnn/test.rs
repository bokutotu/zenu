#[cfg(test)]
mod rnn {
    use zenu_test::assert_mat_eq_epsilon;
    use zenu_test::read_test_case_from_json;

    use crate::device::cpu::Cpu;
    use crate::matrix::Owned;
    use crate::{device::nvidia::Nvidia, dim::DimDyn, matrix::Matrix, nn::rnn::*};

    fn test_json_single(json_path: String, num_layers: usize) {
        let matrix_map = read_test_case_from_json!(json_path);

        let input_weight: Matrix<Owned<f32>, DimDyn, Cpu> =
            matrix_map.get("rnn.weight_ih_l0").unwrap().clone();
        let hidden_weight = matrix_map.get("rnn.weight_hh_l0").unwrap().clone();
        let input_bias = matrix_map.get("rnn.bias_ih_l0").unwrap().clone();
        let hidden_bias = matrix_map.get("rnn.bias_hh_l0").unwrap().clone();

        let input_size = input_weight.shape()[1];
        let output_size = hidden_weight.shape()[0];
        let batch_size = matrix_map.get("input").unwrap().shape()[1];

        let mut config = RNNDescriptor::<f32>::new_rnn_relu(
            false,
            0.0,
            input_size,
            output_size,
            num_layers,
            batch_size,
        );
        // let weight_bytes = config.get_weight_bytes();
        // let param = RNNParameters::new(weight_bytes);
        config.alloc_weight();

        let rnn_weights = RNNWeights::new(
            input_weight,
            hidden_weight,
            Some(input_bias),
            Some(hidden_bias),
        );

        config.load_rnn_weights(vec![rnn_weights]).unwrap();

        let x = matrix_map.get("input").unwrap().clone();
        let x = x.to::<Nvidia>();

        let y = rnn_fwd(x.to_ref(), None, true, &mut config);
        let output = matrix_map.get("output").unwrap().clone();
        let output = output.to::<Nvidia>();
        assert_mat_eq_epsilon!(y.y.to_ref(), output, 1e-5);

        let dy = Matrix::ones_like(&y.y);

        let dx = rnn_bkwd_data(
            x.shape(),
            y.y.to_ref(),
            dy.to_ref(),
            None,
            None,
            &mut config,
        );

        assert_mat_eq_epsilon!(
            dx.dx.to_ref(),
            matrix_map.get("input_grad").unwrap().clone().to::<Nvidia>(),
            1e-5
        );

        let dw = rnn_bkwd_weights(x.to_ref(), None, y.y.to_ref(), &mut config);

        let params = config.store_rnn_weights::<Cpu>(dw.weight as *mut u8);

        for layer_id in 0..num_layers {
            let input_weight = params[layer_id].input_weight();
            let hidden_weight = params[layer_id].hidden_weight();
            let input_bias = params[layer_id].input_bias().unwrap();
            let hidden_bias = params[layer_id].hidden_bias().unwrap();

            let input_weight_expected = matrix_map
                .get(&format!("rnn.weight_ih_l{}_grad", layer_id))
                .unwrap()
                .clone();
            let hidden_weight_expected = matrix_map
                .get(&format!("rnn.weight_hh_l{}_grad", layer_id))
                .unwrap()
                .clone();
            let input_bias_expected = matrix_map
                .get(&format!("rnn.bias_ih_l{}_grad", layer_id))
                .unwrap()
                .clone();
            let hidden_bias_expected = matrix_map
                .get(&format!("rnn.bias_hh_l{}_grad", layer_id))
                .unwrap()
                .clone();

            assert_mat_eq_epsilon!(input_weight, input_weight_expected, 1e-5);
            assert_mat_eq_epsilon!(input_bias, input_bias_expected, 1e-5);
            assert_mat_eq_epsilon!(hidden_bias, hidden_bias_expected, 1e-5);
            assert_mat_eq_epsilon!(hidden_weight, hidden_weight_expected, 1e-5);
        }
    }

    #[test]
    fn test_rnn_seq_len_1() {
        test_json_single(
            "../test_data_json/rnn_fwd_bkwd_single_seq_len_1.json".to_string(),
            1,
        );
    }

    #[test]
    fn test_rnn_seq_len_3() {
        test_json_single(
            "../test_data_json/rnn_fwd_bkwd_single_seq_len_3.json".to_string(),
            1,
        );
    }

    #[test]
    fn test_rnn_seq_len_5() {
        test_json_single("../test_data_json/rnn_fwd_bkwd_single.json".to_string(), 1);
    }
}

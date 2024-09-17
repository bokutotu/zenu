#[cfg(test)]
mod rnn {
    use zenu_test::assert_mat_eq_epsilon;
    use zenu_test::read_test_case_from_json;

    use crate::device::cpu::Cpu;
    use crate::matrix::Owned;
    use crate::{device::nvidia::Nvidia, dim::DimDyn, matrix::Matrix, nn::rnn::*};

    #[expect(clippy::too_many_lines)]
    fn test_json_single(json_path: String, num_layers: usize, bidirectional: bool) {
        let matrix_map = read_test_case_from_json!(json_path);

        let mut weights = Vec::new();
        for layer_id in 0..num_layers {
            let input_weight: Matrix<Owned<f32>, DimDyn, Cpu> = matrix_map
                .get(&format!("rnn.weight_ih_l{layer_id}"))
                .unwrap()
                .clone();
            let hidden_weight = matrix_map
                .get(&format!("rnn.weight_hh_l{layer_id}"))
                .unwrap()
                .clone();
            let input_bias = matrix_map
                .get(&format!("rnn.bias_ih_l{layer_id}"))
                .unwrap()
                .clone();
            let hidden_bias = matrix_map
                .get(&format!("rnn.bias_hh_l{layer_id}"))
                .unwrap()
                .clone();

            let rnn_weights = RNNWeightsMat::new(
                input_weight,
                hidden_weight,
                Some(input_bias),
                Some(hidden_bias),
            );
            weights.push(rnn_weights);

            if bidirectional {
                let input_weight: Matrix<Owned<f32>, DimDyn, Cpu> = matrix_map
                    .get(&format!("rnn.weight_ih_l{layer_id}_reverse"))
                    .unwrap()
                    .clone();
                let hidden_weight = matrix_map
                    .get(&format!("rnn.weight_hh_l{layer_id}_reverse"))
                    .unwrap()
                    .clone();
                let input_bias = matrix_map
                    .get(&format!("rnn.bias_ih_l{layer_id}_reverse"))
                    .unwrap()
                    .clone();
                let hidden_bias = matrix_map
                    .get(&format!("rnn.bias_hh_l{layer_id}_reverse"))
                    .unwrap()
                    .clone();

                let rnn_weights = RNNWeightsMat::new(
                    input_weight,
                    hidden_weight,
                    Some(input_bias),
                    Some(hidden_bias),
                );
                weights.push(rnn_weights);
            }
        }

        let input = matrix_map.get("input").unwrap().clone();
        let output = matrix_map.get("output").unwrap().clone();
        let input_size = input.shape()[2];
        let hidden_size = output.shape()[2] / (if bidirectional { 2 } else { 1 });
        let batch_size = input.shape()[1];

        let mut desc = RNNDescriptor::<f32>::new_rnn_relu(
            bidirectional,
            0.0,
            input_size,
            hidden_size,
            num_layers,
            batch_size,
        );

        let weight_num_elm = desc.get_weight_num_elems();
        let mut weight = Matrix::<Owned<f32>, DimDyn, Nvidia>::alloc([weight_num_elm]);
        desc.load_rnn_weights(weight.to_ref_mut().as_mut_ptr().cast(), weights)
            .unwrap();

        let x = matrix_map.get("input").unwrap().clone();
        let x = x.to::<Nvidia>();

        let hx_num_layers = num_layers * if bidirectional { 2 } else { 1 };
        let hx = Matrix::<_, DimDyn, Nvidia>::zeros([hx_num_layers, batch_size, hidden_size]);

        let y = desc.rnn_fwd(x.to_ref(), Some(hx.to_ref()), weight.to_ref(), true);
        let output = matrix_map.get("output").unwrap().clone();
        let output = output.to::<Nvidia>();
        assert_mat_eq_epsilon!(y.y.to_ref(), output, 1e-5);

        let dy = Matrix::ones_like(&y.y);

        let dhy = Matrix::<_, DimDyn, Nvidia>::zeros_like(&hx);

        let dx = desc.rnn_bkwd_data(
            x.shape(),
            y.y.to_ref(),
            dy.to_ref(),
            Some(hx.to_ref()),
            Some(dhy.to_ref()),
            weight.to_ref(),
        );

        assert_mat_eq_epsilon!(
            dx.dx.to_ref(),
            matrix_map.get("input_grad").unwrap().clone().to::<Nvidia>(),
            1e-5
        );
        let mut dw = desc.rnn_bkwd_weights(x.to_ref(), None, y.y.to_ref());

        let params = desc.store_rnn_weights::<Cpu>(dw.to_ref_mut().as_ptr() as *mut u8);

        for layer_id in 0..num_layers {
            let input_weight = params[layer_id * 2].input_weight();
            let hidden_weight = params[layer_id * 2].hidden_weight();
            let input_bias = params[layer_id * 2].input_bias().unwrap();
            let hidden_bias = params[layer_id * 2].hidden_bias().unwrap();

            let input_weight_expected = matrix_map
                .get(&format!("rnn.weight_ih_l{layer_id}_grad"))
                .unwrap()
                .clone();
            let hidden_weight_expected = matrix_map
                .get(&format!("rnn.weight_hh_l{layer_id}_grad"))
                .unwrap()
                .clone();
            let input_bias_expected = matrix_map
                .get(&format!("rnn.bias_ih_l{layer_id}_grad"))
                .unwrap()
                .clone();
            let hidden_bias_expected = matrix_map
                .get(&format!("rnn.bias_hh_l{layer_id}_grad"))
                .unwrap()
                .clone();

            assert_mat_eq_epsilon!(input_weight, input_weight_expected, 1e-5);
            assert_mat_eq_epsilon!(input_bias, input_bias_expected, 1e-5);
            assert_mat_eq_epsilon!(hidden_bias, hidden_bias_expected, 1e-5);
            assert_mat_eq_epsilon!(hidden_weight, hidden_weight_expected, 1e-5);

            if bidirectional {
                let input_weight = params[layer_id * 2 + 1].input_weight();
                let hidden_weight = params[layer_id * 2 + 1].hidden_weight();
                let input_bias = params[layer_id * 2 + 1].input_bias().unwrap();
                let hidden_bias = params[layer_id * 2 + 1].hidden_bias().unwrap();

                let input_weight_expected = matrix_map
                    .get(&format!("rnn.weight_ih_l{layer_id}_reverse_grad"))
                    .unwrap()
                    .clone();
                let hidden_weight_expected = matrix_map
                    .get(&format!("rnn.weight_hh_l{layer_id}_reverse_grad"))
                    .unwrap()
                    .clone();
                let input_bias_expected = matrix_map
                    .get(&format!("rnn.bias_ih_l{layer_id}_reverse_grad"))
                    .unwrap()
                    .clone();
                let hidden_bias_expected = matrix_map
                    .get(&format!("rnn.bias_hh_l{layer_id}_reverse_grad"))
                    .unwrap()
                    .clone();

                assert_mat_eq_epsilon!(input_weight, input_weight_expected, 1e-5);
                assert_mat_eq_epsilon!(input_bias, input_bias_expected, 1e-5);
                assert_mat_eq_epsilon!(hidden_bias, hidden_bias_expected, 1e-5);
                assert_mat_eq_epsilon!(hidden_weight, hidden_weight_expected, 1e-5);
            }
        }
    }

    #[test]
    fn test_rnn_seq_len_1() {
        test_json_single(
            "../test_data_json/rnn_fwd_bkwd_single_seq_len_1.json".to_string(),
            1,
            false,
        );
    }

    #[test]
    fn test_rnn_seq_len_3() {
        test_json_single(
            "../test_data_json/rnn_fwd_bkwd_single_seq_len_3.json".to_string(),
            1,
            false,
        );
    }

    #[test]
    fn test_rnn_seq_len_5() {
        test_json_single(
            "../test_data_json/rnn_fwd_bkwd_single.json".to_string(),
            1,
            false,
        );
    }

    #[test]
    fn test_rnn_seq_len_5_num_layer_2_bidirectional() {
        test_json_single(
            "../test_data_json/rnn_fwd_bkwd_bidirectional_2_layers.json".to_string(),
            2,
            true,
        );
    }
}

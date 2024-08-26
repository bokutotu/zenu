#[cfg(test)]
mod rnn {
    use zenu_test::assert_mat_eq_epsilon;
    use zenu_test::read_test_case_from_json;

    use crate::device::cpu::Cpu;
    use crate::matrix::Owned;
    use crate::{device::nvidia::Nvidia, dim::DimDyn, matrix::Matrix, nn::rnn::*};

    #[test]
    fn small() {
        let input_size = 2;
        let output_size = 4;
        let batch_size = 1;
        let num_layers = 1;

        let mut config = RNNDescriptor::<f32>::new_rnn_relu(
            false,
            0.0,
            input_size,
            output_size,
            num_layers,
            batch_size,
        );
        let weight_bytes = config.get_weight_bytes();
        let param = RNNParameters::new(weight_bytes);

        let matrix_map =
            read_test_case_from_json!("../test_data_json/matrix/rnn_fwd_bkwd_single.json");

        let input_weight: Matrix<Owned<f32>, DimDyn, Cpu> =
            matrix_map.get("rnn.weight_ih_l0").unwrap().clone();
        let hidden_weight = matrix_map.get("rnn.weight_hh_l0").unwrap().clone();
        let input_bias = matrix_map.get("rnn.bias_ih_l0").unwrap().clone();
        let hidden_bias = matrix_map.get("rnn.bias_hh_l0").unwrap().clone();

        let rnn_weights = RNNWeights::new(
            input_weight,
            hidden_weight,
            Some(input_bias),
            Some(hidden_bias),
        );

        config
            .load_rnn_weights(param.weight, vec![rnn_weights])
            .unwrap();

        let x = matrix_map.get("input").unwrap().clone();
        let x = x.to::<Nvidia>();

        let y = rnn_fwd(x.to_ref(), None, true, &mut config, &param);
        let output = matrix_map.get("output").unwrap().clone();
        let output = output.to::<Nvidia>();
        assert_mat_eq_epsilon!(y.y.to_ref(), output, 1e-5);

        let dy = Matrix::ones_like(&y.y);

        let dx = rnn_bkwd_data(
            x.to_ref(),
            y.y.to_ref(),
            dy.to_ref(),
            None,
            None,
            &mut config,
            &param,
        );

        assert_mat_eq_epsilon!(
            dx.dx.to_ref(),
            matrix_map.get("input_grad").unwrap().clone().to::<Nvidia>(),
            1e-5
        );
    }
}

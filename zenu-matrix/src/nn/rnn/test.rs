#[cfg(test)]
mod rnn {
    use zenu_cuda::runtime::{cuda_copy, ZenuCudaMemCopyKind};
    use zenu_test::assert_mat_eq_epsilon;

    use crate::{
        device::{nvidia::Nvidia, DeviceBase},
        dim::DimDyn,
        matrix::Matrix,
        nn::rnn::*,
    };
    #[test]
    fn small() {
        // rnn.weight_ih_l0
        // [0.38226926, 0.41500396, -0.11713624, 0.45930564, -0.10955179, 0.100895345, -0.24342752, 0.29364133]
        // rnn.weight_hh_l0
        // [0.44077146, -0.36681408, 0.4345981, 0.09357965, 0.36940444, 0.06771529, 0.24109405, -0.0705955, 0.3854429, 0.073904455, -0.23341995, 0.12744915, -0.23036832, -0.058636427, -0.20307916, 0.33168548]
        // rnn.bias_ih_l0
        // [-0.3946851, -0.23050517, -0.14118737, -0.30063623]
        // rnn.bias_hh_l0
        // [0.04719156, -0.49383956, 0.45155454, -0.42473412]
        // Input (flattened): [1.322135090827942, 0.8171897530555725, -0.765838623046875, -0.7506223320960999, 1.3525477647781372, 0.6863219141960144, -0.32775864005088806, 0.7949687242507935, 0.2815195620059967, 0.056163541972637177]
        //
        // Output (flattened): [0.4970550537109375, 0.0, 0.24797554314136505, 0.0, 0.0, 0.0, 0.45223575830459595, 0.0, 0.6509110927581787, 0.0, 0.12587898969650269, 0.0, 0.19873936474323273, 0.0, 0.6479887366294861, 0.0, 0.15264412760734558, 0.0, 0.2105420082807541, 0.0]
        let input_size = 2;
        let output_size = 4;
        let batch_size = 1;
        let seq_length = 5;
        let num_layers = 1;

        let config = RNNConfig::<f32>::new_rnn_relu(
            false,
            0.0,
            input_size,
            output_size,
            num_layers,
            batch_size,
        );
        let weight_bytes = config.get_weight_bytes();
        let weight_ptr = Nvidia::alloc(weight_bytes).unwrap();

        let rnn_params = &config.config.get_rnn_params(weight_ptr as *mut _)[0];

        let input_weight = vec![
            0.38226926,
            0.41500396,
            -0.11713624,
            0.45930564,
            -0.10955179,
            0.100895345,
            -0.24342752,
            0.29364133,
        ];
        let input_bias = vec![-0.3946851, -0.23050517, -0.14118737, -0.30063623];

        let hidden_weight = vec![
            0.44077146,
            -0.36681408,
            0.4345981,
            0.09357965,
            0.36940444,
            0.06771529,
            0.24109405,
            -0.0705955,
            0.3854429,
            0.073904455,
            -0.23341995,
            0.12744915,
            -0.23036832,
            -0.058636427,
            -0.20307916,
            0.33168548,
        ];
        let hidden_bias = vec![0.04719156, -0.49383956, 0.45155454, -0.42473412];

        cuda_copy(
            rnn_params.input_weight.ptr as *mut f32,
            input_weight.as_ptr(),
            input_size * output_size,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        cuda_copy(
            rnn_params.input_bias.ptr as *mut f32,
            input_bias.as_ptr(),
            output_size,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        cuda_copy(
            rnn_params.hidden_weight.ptr as *mut f32,
            hidden_weight.as_ptr(),
            output_size * output_size,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        cuda_copy(
            rnn_params.hidden_bias.ptr as *mut f32,
            hidden_bias.as_ptr(),
            output_size,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        let input = Matrix::from_vec(
            vec![
                1.322135090827942,
                0.8171897530555725,
                -0.765838623046875,
                -0.7506223320960999,
                1.3525477647781372,
                0.6863219141960144,
                -0.32775864005088806,
                0.7949687242507935,
                0.2815195620059967,
                0.056163541972637177,
            ],
            [seq_length, batch_size, input_size],
        );

        let params = RNNParameters { weight: weight_ptr };

        let hx = Matrix::<_, DimDyn, _>::zeros([1, output_size]);
        let output = rnn_fwd(input.to_ref(), Some(hx.to_ref()), true, config, params);
        let y = output.y.to_ref();
        let ans = vec![
            0.4970550537109375,
            0.0,
            0.24797554314136505,
            0.0,
            0.0,
            0.0,
            0.45223575830459595,
            0.0,
            0.6509110927581787,
            0.0,
            0.12587898969650269,
            0.0,
            0.19873936474323273,
            0.0,
            0.6479887366294861,
            0.0,
            0.15264412760734558,
            0.0,
            0.2105420082807541,
            0.0,
        ];
        let ans = Matrix::<_, DimDyn, _>::from_vec(ans, [seq_length, batch_size, output_size]);
        assert_mat_eq_epsilon!(y, ans, 1e-6);
    }
}

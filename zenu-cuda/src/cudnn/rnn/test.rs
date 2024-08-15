#[cfg(test)]
mod rnn {
    use crate::{
        cudnn::rnn::{
            executor::RNNExecutor, RNNAlgo, RNNBias, RNNCell, RNNConfig, RNNDataLayout, RNNMathType,
        },
        runtime::{cuda_copy, cuda_malloc, cuda_malloc_bytes, ZenuCudaMemCopyKind},
    };

    #[test]
    fn rnn() {
        let input_size = 2;
        let output_size = 4;
        let batch_size = 1;
        let seq_length = 5;
        let num_layers = 1;
        let config = RNNConfig::<f32>::new(
            RNNAlgo::Standard,
            RNNCell::RNNRelu,
            RNNBias::DoubleBias,
            false,
            RNNMathType::TensorOp,
            None,
            input_size,
            output_size,
            num_layers,
            batch_size,
        );

        let weight_bytes = config.weights_size;

        let weight_ptr = cuda_malloc_bytes(weight_bytes).unwrap() as *mut f32;

        let rnn_params = &config.get_rnn_params(weight_ptr)[0];

        let input_weight = vec![
            -0.00374341,
            0.2682218,
            -0.41152257,
            -0.3679695,
            -0.19257718,
            0.13407868,
            -0.00990659,
            0.39644474,
        ];
        let input_bias = vec![-0.08059168, 0.05290705, 0.4527381, -0.46383518];

        let hidden_weight = vec![
            -0.044372022,
            0.13230628,
            -0.15110654,
            -0.098282695,
            -0.47767425,
            -0.33114105,
            -0.20611155,
            0.018521786,
            0.1976676,
            0.3000114,
            -0.33897054,
            -0.21773142,
            0.18160856,
            0.41519397,
            -0.10290009,
            0.37415588,
        ];
        let hidden_bias = vec![-0.31476897, -0.12658262, -0.19489998, 0.4320004];

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

        let input = vec![
            -0.5663175, 0.37311465, -0.8919953, -1.5091077, 0.37039354, 1.4565026, 0.9398099,
            0.7748488, 0.19186942, 1.2637948,
        ];

        let input_gpu = cuda_malloc::<f32>(10).unwrap();
        cuda_copy(
            input_gpu,
            input.as_ptr(),
            input_size * batch_size * seq_length,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        let exe = RNNExecutor::new(
            &config,
            seq_length,
            batch_size,
            &[seq_length],
            RNNDataLayout::SeqMajorPacked,
            0_f32,
            true,
        );

        let output_gpu = cuda_malloc::<f32>(20).unwrap();

        let workspace = exe.workspace.workspace_size;
        let workspace = cuda_malloc_bytes(workspace).unwrap();

        let reserve = exe.workspace.reserve_size;
        let reserve = cuda_malloc_bytes(reserve).unwrap();

        exe.fwd(
            input_gpu,
            output_gpu,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            weight_ptr,
            workspace as *mut f32,
            reserve as *mut f32,
        );

        let mut output: Vec<f32> = vec![0.0; 20];
        cuda_copy(
            output.as_mut_ptr(),
            output_gpu,
            output.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();

        let ans = vec![
            0.0,
            0.022082046,
            0.41692466,
            0.12169483,
            0.0,
            0.7577149,
            0.066079825,
            0.0,
            0.08418392,
            0.0,
            0.58671874,
            0.8497178,
            0.0,
            0.0,
            0.0,
            0.53888166,
            0.0,
            0.0,
            0.27300492,
            0.6689149,
        ];

        for i in 0..output.len() {
            assert!((output[i] - ans[i]).abs() < 1e-6);
        }
    }
}

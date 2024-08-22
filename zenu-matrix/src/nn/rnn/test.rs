#[cfg(test)]
mod rnn {
    use std::collections::HashMap;

    use zenu_cuda::runtime::{cuda_copy, ZenuCudaMemCopyKind};
    use zenu_test::assert_mat_eq_epsilon;
    use zenu_test::read_test_case_from_json;

    use crate::device::cpu::Cpu;
    use crate::matrix::Owned;
    use crate::{
        device::{nvidia::Nvidia, Device, DeviceBase},
        dim::DimDyn,
        matrix::Matrix,
        nn::rnn::*,
    };
    #[test]
    fn small() {
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

        let matrix_map =
            read_test_case_from_json("../test_data_json/matrix/rnn_fwd_bkwd_single.json");
    }
}

use std::collections::HashMap;

use zenu_matrix::{
    device::cpu::Cpu,
    dim::DimDyn,
    matrix::{Matrix, Owned},
};
use zenu_test::read_test_case_from_json_val;

fn lstm(json_path: String, num_layers: usize, bidirectional: bool) {
    let matrix_mat: HashMap<String, Matrix<Owned<f32>, DimDyn, Cpu>> =
        read_test_case_from_json_val!(json_path);

    // let mut weights = Vec::new();
    // for layer_id in 0..num_layers {
    //     let input_gate_x = matrix_mat
    //         .get(&format!("lstm.input_gate_x_l{layer_id}"))
    //         .unwrap()
    //         .clone();
    // }
}

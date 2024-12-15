use interface::{ConvBkwdDataConfig, ConvBkwdFilterConfig, ConvFwdConfig};

use crate::{
    device::Device,
    dim::DimDyn,
    matrix::{Matrix, Ref},
    num::Num,
};

pub mod interface;

mod cpu;
mod utils;

#[cfg(feature = "nvidia")]
mod nvidia;

#[expect(clippy::module_name_repetitions)]
pub fn conv_fwd<T: Num, D: Device>(
    input: Matrix<Ref<&T>, DimDyn, D>,
    weight: Matrix<Ref<&T>, DimDyn, D>,
    output: Matrix<Ref<&mut T>, DimDyn, D>,
    config: &mut ConvFwdConfig<T>,
) {
    D::conv_fwd(input, weight, output, config);
}

#[expect(clippy::module_name_repetitions)]
pub fn conv_bkwd_data<T: Num, D: Device>(
    dy: Matrix<Ref<&T>, DimDyn, D>,
    filter: Matrix<Ref<&T>, DimDyn, D>,
    dx: Matrix<Ref<&mut T>, DimDyn, D>,
    config: &mut ConvBkwdDataConfig<T>,
) {
    D::conv_bkwd_data(dy, filter, dx, config);
}

#[expect(clippy::module_name_repetitions)]
pub fn conv_bkwd_weight<T: Num, D: Device>(
    dy: Matrix<Ref<&T>, DimDyn, D>,
    x: Matrix<Ref<&T>, DimDyn, D>,
    dw: Matrix<Ref<&mut T>, DimDyn, D>,
    config: &mut ConvBkwdFilterConfig<T>,
) {
    D::conv_bkwd_filter(dy, x, dw, config);
}

#[cfg(test)]
mod conv_test {
    use std::collections::HashMap;
    use zenu_test::{assert_mat_eq_epsilon, read_test_case_from_json, run_mat_test};

    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
        nn::conv::{conv_fwd, interface::ConvFwdConfig},
    };

    fn load_test_case<D: Device>(path: &str) -> HashMap<String, Matrix<Owned<f32>, DimDyn, D>> {
        let map = read_test_case_from_json!(path);
        map.iter()
            .map(|(key, value)| (key.clone(), value.clone().to()))
            .collect()
    }

    fn small_test<D: Device>() {
        let matrix_map = load_test_case::<D>("../test_data_json/conv2d.json");

        let x = matrix_map.get("input").unwrap().clone();
        let w = matrix_map.get("filter").unwrap().clone();
        let y = matrix_map.get("output").unwrap().clone();

        let mut y_hat = Matrix::zeros(y.shape());

        let mut config = ConvFwdConfig::new(
            x.shape_stride(),
            w.shape_stride(),
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
        );

        conv_fwd(x.to_ref(), w.to_ref(), y_hat.to_ref_mut(), &mut config);

        assert_mat_eq_epsilon!(y_hat, y, 1e-4);
    }

    run_mat_test!(small_test, small_test_cpu, small_test_gpu);
}

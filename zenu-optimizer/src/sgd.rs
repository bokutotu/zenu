use zenu_autograd::Variable;
use zenu_layer::Parameters;
use zenu_matrix::{device::Device, num::Num};

use crate::Optimizer;

pub struct SGD<T: Num, D: Device> {
    pub learning_rate: T,
    _device: std::marker::PhantomData<D>,
}

impl<T: Num, D: Device> SGD<T, D> {
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            _device: std::marker::PhantomData,
        }
    }
}

impl<T: Num, D: Device, P: Parameters<T, D>> Optimizer<T, D, P> for SGD<T, D> {
    fn update(&self, parameters: &P) {
        let weights = parameters.weights();
        let biases = parameters.biases();
        let mut parameters = Vec::new();
        for (_, weight) in weights.iter() {
            if let Some(grad) = weight.get_grad() {
                parameters.push(grad);
            }
        }
        for (_, bias) in biases.iter() {
            if let Some(grad) = bias.get_grad() {
                parameters.push(grad);
            }
        }
        for parameter in parameters {
            let grad = parameter.clone().get_grad().unwrap();
            let grad = grad.get_data();
            let update_data = grad.to_ref() * self.learning_rate;

            let mut data = parameter.get_data_mut();
            let mut data = data.to_ref_mut();
            data -= update_data;
        }
    }
}

#[cfg(test)]
mod sgd {
    use zenu_autograd::creator::from_vec::from_vec;
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_mat_eq_epsilon, run_test};

    use crate::Optimizer;

    use super::SGD;

    // #[test]
    fn simple_test<D: Device>() {
        let variable = from_vec::<f32, _, D>(vec![1., 2., 3., 4., 5., 6.], [3, 2]);
        variable.set_grad(from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]));
        let sgd = SGD::new(1.);
        sgd.update(&[variable.clone()]);
        let data = variable.get_data();
        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![0., 0., 0., 0., 0., 0.], [3, 2]);
        assert_mat_eq_epsilon!(data, ans, 1e-6);
    }
    run_test!(simple_test, simple_test_cpu, simple_test_nvidia);
}

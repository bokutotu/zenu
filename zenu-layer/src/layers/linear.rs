use std::collections::HashMap;

use crate::{Module, Parameters};
use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::{
    creator::{rand::normal, zeros::zeros},
    functions::{matmul::matmul, transpose::transpose},
    Variable,
};
use zenu_matrix::{device::Device, num::Num};

pub struct Linear<T: Num, D: Device> {
    in_features: usize,
    out_features: usize,
    pub weight: Variable<T, D>,
    pub bias: Option<Variable<T, D>>,
}

impl<T: Num, D: Device> Module<T, D> for Linear<T, D> {
    type Input = Variable<T, D>;
    type Output = Variable<T, D>;
    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
        let weight_t = transpose(self.weight.clone());
        let output = matmul(input, weight_t);
        if let Some(bias) = &self.bias {
            output.set_name("linear.intermediate_output");
            output + bias.clone()
        } else {
            output
        }
    }
}

impl<T: Num, D: Device> Parameters<T, D> for Linear<T, D> {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        let mut weights = HashMap::new();
        weights.insert("linear.weight".to_string(), self.weight.clone());
        weights
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        let mut biases = HashMap::new();
        if let Some(bias) = &self.bias {
            biases.insert("linear.bias".to_string(), bias.clone());
        }
        biases
    }
}

impl<T: Num, D: Device> Linear<T, D> {
    #[must_use]
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let weight = normal(T::zero(), T::one(), None, [out_features, in_features]);
        weight
            .get_data_mut()
            .to_ref_mut()
            .div_scalar_assign(T::from_usize(in_features).sqrt());
        let bias = if use_bias {
            let bias = zeros([out_features]);
            bias.set_name("linear.bias");
            bias.set_is_train(true);
            Some(bias)
        } else {
            None
        };

        weight.set_is_train(true);
        weight.set_name("linear.weight");

        Self {
            in_features,
            out_features,
            weight,
            bias,
        }
    }

    #[must_use]
    pub fn to<Dout: Device>(self) -> Linear<T, Dout> {
        Linear {
            in_features: self.in_features,
            out_features: self.out_features,
            weight: self.weight.to(),
            bias: self.bias.map(|b| b.to()),
        }
    }
}

// #[cfg(test)]
// mod linear {
//     use zenu_autograd::creator::rand::normal;
//     use zenu_matrix::{device::Device, dim::DimTrait, operation::mul::matmul};
//     use zenu_test::{assert_mat_eq_epsilon, assert_val_eq, run_test};
//
//     use crate::{Module, StateDict};
//
//     use super::Linear;
//
//     fn with_bias<D: Device>() {
//         let layer = Linear::<f32, D>::new(3, 2, true);
//         let input = normal::<_, _, D>(0., 1., None, [5, 3]);
//         let output = layer.call(input.clone());
//         assert_eq!(output.get_data().shape().slice(), [5, 2]);
//
//         let parameters = layer.to_json();
//
//         let ans = matmul(
//             &input.get_data().to_ref(),
//             &layer.weight.get_data().to_ref(),
//         ) + &layer.bias.unwrap().get_data().to_ref();
//
//         assert_val_eq!(output.clone(), ans, 1e-4);
//
//         let new_layer = Linear::<f32, D>::from_json(&parameters);
//         let new_output = new_layer.call(input.clone());
//
//         assert_mat_eq_epsilon!(output.get_data(), new_output.get_data(), 1e-4);
//     }
//     run_test!(with_bias, with_bias_cpu, with_bias_gpu);
//
//     fn without_bias<D: Device>() {
//         let layer = Linear::<f32, D>::new(3, 2, false);
//         let input = normal::<_, _, D>(0., 1., None, [5, 3]);
//         let output = layer.call(input.clone());
//         assert_eq!(output.get_data().shape().slice(), [5, 2]);
//
//         let parameters = layer.to_json();
//         let weight = layer.weight.clone();
//
//         let ans = matmul(&input.get_data().to_ref(), &weight.get_data().to_ref());
//         assert_val_eq!(output.clone(), ans, 1e-4);
//
//         let new_layer = Linear::<f32, D>::from_json(&parameters);
//         let new_output = new_layer.call(input.clone());
//
//         assert_mat_eq_epsilon!(output.get_data(), new_output.get_data(), 1e-4);
//     }
//     run_test!(without_bias, without_bias_cpu, without_bias_gpu);
// }

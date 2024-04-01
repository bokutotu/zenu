use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    constructor::{ones::Ones, zeros::Zeros},
    dim::DimTrait,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::OwnedMatrixDyn,
    num::Num,
    operation::{
        basic_operations::{MatrixAddAssign, MatrixMulAssign, MatrixSqrt},
        copy_from::CopyFrom,
        mean::Mean,
        reshape::Reshape,
        transpose::TransposeInplace,
        var::Variance,
    },
};

use crate::{creator::zeros::zeros, is_train, Function, Variable, VariableWeak};

use super::{reshape::reshape, sum::sum, transpose::transpose_by_index};

struct BatchNorm<T: Num> {
    mean: Variable<T>,
    variance: Variable<T>,
    decay: Variable<T>,
    epsilon: Variable<T>,
    inv_std: Variable<T>,
    gamma: Variable<T>,
    beta: Variable<T>,
    input: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> BatchNorm<T> {
    fn new(
        mean: Variable<T>,
        variance: Variable<T>,
        decay: Variable<T>,
        epsilon: Variable<T>,
        inv_std: Variable<T>,
        gamma: Variable<T>,
        beta: Variable<T>,
        input: Variable<T>,
        output: Variable<T>,
    ) -> Self {
        let output = output.downgrade();
        Self {
            mean,
            variance,
            decay,
            epsilon,
            inv_std,
            gamma,
            beta,
            input,
            output,
        }
    }
}

impl<T: Num> Function<T> for BatchNorm<T> {
    fn forward(&self) {
        let input_mat = self.input.get_data();
        let input_mat = if input_mat.shape().len() == 4 {
            let channel = input_mat.shape()[1];
            let num_elm = input_mat.shape().num_elm();
            let input_mat = input_mat.transepose_by_index(&[0, 2, 3, 1]);
            input_mat.reshape_new_matrix([num_elm / channel, channel])
        } else {
            input_mat
        };

        let xc;

        if is_train() {
            let mean = input_mat.mean(Some(0), false);
            let var = input_mat.variance(Some(0), false);
            let var_eps = var.to_view() + self.epsilon.get_data();
            let mut zeros = OwnedMatrixDyn::zeros_like(var_eps.to_view());
            zeros.to_view_mut().sqrt(var_eps);
            let inv_std = OwnedMatrixDyn::ones(var.shape()) / zeros;
            xc = (input_mat.to_view() - mean.to_view()) * inv_std.to_view();
            let m = input_mat.shape().num_elm() / self.gamma.get_data().shape().num_elm();
            let s = if m - 1 > 1 { m - 1 } else { 1 };
            let adjust = m / s;
            self.mean.get_data_mut().mul_assign(self.decay.get_data());
            self.mean.get_data_mut().to_view_mut().add_assign(
                mean.to_view()
                    * (OwnedMatrixDyn::ones(self.decay.get_data().shape()) - self.decay.get_data()),
            );
            self.variance
                .get_data_mut()
                .mul_assign(self.decay.get_data());
            self.variance.get_data_mut().to_view_mut().add_assign(
                var.to_view()
                    * (OwnedMatrixDyn::ones(self.decay.get_data().shape()) - self.decay.get_data())
                    * T::from_usize(adjust),
            );
            self.inv_std
                .get_data_mut()
                .to_view_mut()
                .copy_from(&inv_std.to_view());
        } else {
            let inv_std = OwnedMatrixDyn::ones(self.variance.get_data().shape())
                / (self.variance.get_data() + self.epsilon.get_data());
            xc = (input_mat.to_view() - self.mean.get_data().to_view()) * inv_std.to_view();
        }
        let output =
            self.gamma.get_data().to_view() * xc.to_view() + self.beta.get_data().to_view();

        let output = if input_mat.shape().len() == 4 {
            let channel = input_mat.shape()[1];
            let num_elm = input_mat.shape().num_elm();
            let output = output.reshape_new_matrix([num_elm / channel, channel]);
            output.transpose_by_index_inplace(&[0, 3, 1, 2])
        } else {
            output
        };

        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_view_mut()
            .copy_from(&output);
    }

    fn backward(&self) {
        if !is_train() {
            panic!("backward is called in inference mode");
        }
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        let output_grad = if output_grad.get_data().shape().len() == 4 {
            let channel = output_grad.get_data().shape()[1];
            let num_elm = output_grad.get_data().shape().num_elm();
            let output_grad = transpose_by_index(output_grad, vec![0, 2, 3, 1]);
            reshape(output_grad, &[num_elm / channel, channel])
        } else {
            output_grad
        };
        let batch_size = output_grad.get_data().shape()[0];
        let input = if self.input.get_data().shape().len() == 4 {
            let channel = self.input.get_data().shape()[1];
            let num_elm = self.input.get_data().shape().num_elm();
            let input = transpose_by_index(self.input.clone(), vec![0, 2, 3, 1]);
            reshape(input, &[num_elm / channel, channel])
        } else {
            self.input.clone()
        };
        let mean = sum(input.clone(), 0, false) / Variable::from(T::from_usize(batch_size));
        let xc = (input - mean) * self.inv_std.clone();
        let beta_grad = sum(output_grad.clone(), 0, false);
        let gamma_grad = sum(xc.clone() * output_grad.clone(), 0, false);

        // let input_grad = output_grad.clone()
        //     - beta_grad.clone() / Variable::from(T::from_usize(batch_size))
        //     - xc.clone() * gamma_grad.clone() / Variable::from(T::from_usize(batch_size));
        // gbeta / batch_size
        let beta_grad_batch_size = beta_grad.clone() / Variable::from(T::from_usize(batch_size));
        let xc_gamma_grad_batch_size =
            xc.clone() * gamma_grad.clone() / Variable::from(T::from_usize(batch_size));
        let input_grad = output_grad.clone() - beta_grad_batch_size - xc_gamma_grad_batch_size;
        let input_grad = input_grad * self.gamma.clone() * self.inv_std.clone();

        if self.input.get_data().shape().len() == 4 {
            let channel = input_grad.get_data().shape()[1];
            let num_elm = input_grad.get_data().shape().num_elm();
            let input_grad = reshape(input_grad, &[num_elm / channel, channel]);
            let input_grad = transpose_by_index(input_grad, vec![0, 3, 1, 2]);
            self.input.set_grad(input_grad);
        } else {
            self.input.set_grad(input_grad);
        }
        self.gamma.set_grad(gamma_grad);
        self.beta.set_grad(beta_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![
            self.mean.clone(),
            self.variance.clone(),
            self.decay.clone(),
            self.epsilon.clone(),
            self.inv_std.clone(),
            self.input.clone(),
        ]
    }
}

pub fn batch_norm<T: Num>(
    mean: Variable<T>,
    variance: Variable<T>,
    decay: Variable<T>,
    epsilon: Variable<T>,
    gamma: Variable<T>,
    beta: Variable<T>,
    input: Variable<T>,
) -> Variable<T> {
    let output_shape = input.get_data().shape();
    let output = zeros(output_shape);
    let inv_std = zeros(variance.get_data().shape());
    let batch_norm = BatchNorm::new(
        mean,
        variance,
        decay,
        epsilon,
        inv_std,
        gamma,
        beta,
        input,
        output.clone(),
    );
    batch_norm.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(batch_norm))));
    output
}

#[cfg(test)]
mod batch_norm {
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use super::batch_norm;

    #[test]
    fn batch_norm_medium() {
        let input = (1..31).map(|x| x as f64).collect::<Vec<f64>>();
        let input = OwnedMatrixDyn::from_vec(input, &[10, 3]);
        let mean = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let var = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let decay = OwnedMatrixDyn::from_vec(vec![0.9], &[]);
        let epsilon = OwnedMatrixDyn::from_vec(vec![1e-5], &[]);
        let gamma = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let delta = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let input = crate::Variable::from(input);
        let mean = crate::Variable::from(mean);
        let var = crate::Variable::from(var);
        let decay = crate::Variable::from(decay);
        let epsilon = crate::Variable::from(epsilon);
        let gamma = crate::Variable::from(gamma);
        let beta = crate::Variable::from(delta);
        let output = batch_norm(
            mean,
            var,
            decay,
            epsilon,
            gamma.clone(),
            beta.clone(),
            input.clone(),
        );
        output.backward();
        let ans = vec![
            -0.5666988,
            -1.1333976,
            -1.70009639,
            -0.21854351,
            -0.43708702,
            -0.65563053,
            0.12961178,
            0.25922356,
            0.38883534,
            0.47776707,
            0.95553413,
            1.4333012,
            0.82592236,
            1.65184471,
            2.47776707,
            1.17407764,
            2.34815529,
            3.52223293,
            1.52223293,
            3.04446587,
            4.5666988,
            1.87038822,
            3.74077644,
            5.61116466,
            2.21854351,
            4.43708702,
            6.65563053,
            2.5666988,
            5.1333976,
            7.70009639,
        ];
        let ans = OwnedMatrixDyn::from_vec(ans, &[10, 3]);
        assert!((ans - output.get_data()).asum() < 1e-6);
        let input_grad_ans = vec![
            4.037174091273418e-18,
            8.074348182546835e-18,
            1.2111522273820252e-17,
            3.1400242932126583e-18,
            6.2800485864253165e-18,
            9.420072879637973e-18,
            2.2428744951518985e-18,
            4.485748990303797e-18,
            6.728623485455695e-18,
            1.3457246970911395e-18,
            2.691449394182279e-18,
            4.037174091273418e-18,
            4.485748990303797e-19,
            8.971497980607594e-19,
            1.345724697091139e-18,
            -4.485748990303797e-19,
            -8.971497980607594e-19,
            -1.345724697091139e-18,
            -1.3457246970911395e-18,
            -2.691449394182279e-18,
            -4.037174091273418e-18,
            -2.2428744951518985e-18,
            -4.485748990303797e-18,
            -6.728623485455695e-18,
            -3.1400242932126583e-18,
            -6.2800485864253165e-18,
            -9.420072879637973e-18,
            -4.037174091273418e-18,
            -8.074348182546835e-18,
            -1.2111522273820252e-17,
        ];
        let input_grad_ans = OwnedMatrixDyn::from_vec(input_grad_ans, &[10, 3]);
        let gamma_grad = vec![
            2.220446049250313e-16,
            2.220446049250313e-16,
            2.220446049250313e-16,
        ];
        let gamma_grad = OwnedMatrixDyn::from_vec(gamma_grad, &[3]);
        let beta_grad = vec![10., 10., 10.];
        let beta_grad = OwnedMatrixDyn::from_vec(beta_grad, &[3]);
        assert!((input_grad_ans - input.get_grad().unwrap().get_data()).asum() < 1e-25);
        assert!((gamma_grad - gamma.get_grad().unwrap().get_data()).asum() < 1e-25);
        assert!((beta_grad - beta.get_grad().unwrap().get_data()).asum() < 1e-25);
    }
}

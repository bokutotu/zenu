use std::{cell::RefCell, rc::Rc};

use zenu_autograd::{creator::zeros::zeros_like, Variable};
use zenu_matrix::{device::Device, num::Num};

use crate::Optimizer;

pub struct Adam<T: Num, D: Device> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    step: Rc<RefCell<T>>,
    m: Vec<Variable<T, D>>,
    v: Vec<Variable<T, D>>,
}

impl<T: Num, D: Device> Optimizer<T, D> for Adam<T, D> {
    fn update(&self, parameters: &[Variable<T, D>]) {
        let step = *self.step.borrow();
        let step = step + T::one();
        *self.step.borrow_mut() = step;

        let beta1_t = self.beta1.powf(step);
        let beta2_t = self.beta2.powf(step);

        for ((parameter, m), v) in parameters.iter().zip(&self.m).zip(&self.v) {
            let grad = parameter.get_grad().unwrap();
            let grad = grad.get_data();

            let mut v = v.get_data_mut();
            let mut v = v.to_ref_mut();
            let mut m = m.get_data_mut();
            let mut m = m.to_ref_mut();

            m *= self.beta1;
            m += grad.to_ref() * (T::one() - self.beta1);

            v *= self.beta2;
            v += grad.to_ref() * grad.to_ref() * (T::one() - self.beta2);

            let m_hat = m / (T::one() - beta1_t);
            let v_hat = v / (T::one() - beta2_t);

            let mut parameter_data = parameter.get_data_mut();
            let mut parameter_data = parameter_data.to_ref_mut();
            parameter_data -= m_hat / (v_hat.sqrt() + self.epsilon) * self.learning_rate;
        }
    }
}

impl<T: Num, D: Device> Adam<T, D> {
    pub fn new(
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        parameters: &[Variable<T, D>],
    ) -> Self {
        let m = parameters
            .iter()
            .map(|parameter| zeros_like(parameter))
            .collect();
        let v = parameters
            .iter()
            .map(|parameter| zeros_like(parameter))
            .collect();
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            step: Rc::new(RefCell::new(T::zero())),
            m,
            v,
        }
    }
}

#[cfg(test)]
mod adam_tests {
    use zenu_autograd::{
        creator::from_vec::from_vec, functions::matmul::matmul, loss::mse::mean_squared_error,
        Variable,
    };
    use zenu_matrix::{device::Device, dim::DimDyn, matrix::Matrix};
    use zenu_test::{assert_val_eq, run_test};

    use crate::Optimizer;

    use super::Adam;

    fn simple_function<D: Device>(
        x: Variable<f64, D>,
        weight1: Variable<f64, D>,
        weight2: Variable<f64, D>,
    ) -> Variable<f64, D> {
        let x = matmul(x, weight1);
        matmul(x, weight2)
    }

    #[expect(clippy::needless_pass_by_value, clippy::type_complexity)]
    fn adam_apply<D: Device>(
        adam: &Adam<f64, D>,
        forward_func: fn(Variable<f64, D>, Variable<f64, D>, Variable<f64, D>) -> Variable<f64, D>,
        input: Variable<f64, D>,
        target: Variable<f64, D>,
        weight1: Variable<f64, D>,
        weight2: Variable<f64, D>,
    ) {
        let output = forward_func(input.clone(), weight1.clone(), weight2.clone());
        let loss = mean_squared_error(target, output);
        loss.backward();
        adam.update(&[weight1.clone(), weight2.clone()]);
        loss.clear_grad();
    }

    #[expect(clippy::unreadable_literal)]
    fn small_2_times<D: Device>() {
        // Initial weights:
        // Weight1: 10.000000
        // Weight2: 10.000000
        //
        // Iteration 1:
        // Input: 1.000000
        // Target: 6.000000
        // Weight1: 9.900000
        // Weight2: 9.900000
        // Loss: 8836.000000
        //
        // Iteration 2:
        // Input: 1.100000
        // Target: 6.600000
        // Weight1: 9.799901
        // Weight2: 9.799901
        // Loss: 10243.665039
        let ans_weight_1 = from_vec::<f64, _, D>(vec![2.], [1, 1]);
        let ans_weight_2 = from_vec::<f64, _, D>(vec![3.], [1, 1]);

        let weight_1 = from_vec::<f64, _, D>(vec![10.], [1, 1]);
        let weight_2 = from_vec::<f64, _, D>(vec![10.], [1, 1]);

        let adam = Adam::new(0.1, 0.9, 0.999, 1e-8, &[weight_1.clone(), weight_2.clone()]);

        // iter 1
        let input = from_vec::<f64, _, D>(vec![1.], [1, 1]);
        let target = simple_function(input.clone(), ans_weight_1.clone(), ans_weight_2.clone());
        adam_apply(
            &adam,
            simple_function,
            input,
            target,
            weight_1.clone(),
            weight_2.clone(),
        );
        let iter_1_weight_1 = Matrix::<_, DimDyn, D>::from_vec(vec![9.9], [1, 1]);
        let iter_1_weight_2 = Matrix::<_, DimDyn, D>::from_vec(vec![9.9], [1, 1]);
        assert_val_eq!(weight_1.clone(), iter_1_weight_1, 1e-6);
        assert_val_eq!(weight_2.clone(), iter_1_weight_2, 1e-6);

        // iter 2
        let input = from_vec::<f64, _, D>(vec![1.1], [1, 1]);
        let target = simple_function(input.clone(), ans_weight_1.clone(), ans_weight_2.clone());
        adam_apply(
            &adam,
            simple_function,
            input,
            target,
            weight_1.clone(),
            weight_2.clone(),
        );
        let iter_2_weight_1 = Matrix::<_, DimDyn, D>::from_vec(vec![9.799901], [1, 1]);
        let iter_2_weight_2 = Matrix::<_, DimDyn, D>::from_vec(vec![9.799901], [1, 1]);
        assert_val_eq!(weight_1.clone(), iter_2_weight_1, 2e-4);
        assert_val_eq!(weight_2.clone(), iter_2_weight_2, 2e-4);
    }
    run_test!(small_2_times, small_2_times_cpu, small_2_times_gpu);
}

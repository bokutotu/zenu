use std::{cell::RefCell, rc::Rc};

use zenu_autograd::Variable;
use zenu_matrix::{
    constructor::zeros::Zeros,
    matrix::{MatrixBase, ToOwnedMatrix, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::OwnedMatrixDyn,
    num::Num,
    operation::{
        basic_operations::{MatrixAddAssign, MatrixPowf, MatrixSqrt, MatrixSubAssign},
        copy_from::CopyFrom,
    },
};

use crate::Optimizer;

pub struct Adam<T: Num> {
    pub alpha: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    m: Rc<RefCell<Vec<OwnedMatrixDyn<T>>>>,
    v: Rc<RefCell<Vec<OwnedMatrixDyn<T>>>>,
}

impl<T: Num> Adam<T> {
    pub fn new(alpha: T, beta1: T, beta2: T, epsilon: T) -> Self {
        Self {
            alpha,
            beta1,
            beta2,
            epsilon,
            m: Rc::new(RefCell::new(Vec::new())),
            v: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn init_m(&self, parameters: &[Variable<T>]) {
        let m = &mut *self.m.borrow_mut();
        m.clear();
        for p in parameters {
            m.push(OwnedMatrixDyn::zeros(p.get_data().shape()));
        }
    }

    fn init_v(&self, parameters: &[Variable<T>]) {
        let v = &mut *self.v.borrow_mut();
        v.clear();
        for p in parameters {
            v.push(OwnedMatrixDyn::zeros(p.get_data().shape()));
        }
    }

    fn update_m(&self, parameters: &[Variable<T>]) {
        let m = &mut *self.m.borrow_mut();
        for (m, p) in m.iter_mut().zip(parameters) {
            let mut p_g = p.get_grad().unwrap().get_data();
            let m_c = m.to_owned_matrix();
            p_g.sub_assign(m_c);
            m.to_view_mut().add_assign(p_g * (T::one() - self.beta1));
        }
    }

    fn update_v(&self, parameters: &[Variable<T>]) {
        let v = &mut *self.v.borrow_mut();
        for (v, p) in v.iter_mut().zip(parameters) {
            let p_g = p.get_grad().unwrap().get_data();
            let v_c = v.to_owned_matrix();
            let mut p_g = p_g.to_view() * p_g.to_view();
            p_g.sub_assign(v_c);
            v.to_view_mut().add_assign(p_g * (T::one() - self.beta2));
        }
    }

    fn update_parameters(&self, parameters: &[Variable<T>]) {
        let m = self.m.borrow();
        let v = self.v.borrow();
        for ((p, m), v) in parameters.iter().zip(m.iter()).zip(v.iter()) {
            let m_hat = m.clone() / (T::one() - self.beta1);
            let mut v_hat = v.clone() / (T::one() - self.beta2);
            let v_hat_c = v_hat.to_owned_matrix();
            v_hat.to_view_mut().sqrt(v_hat_c);

            let diff_p = m_hat * self.beta2 / (v_hat + self.epsilon);
            p.get_data_mut()
                .to_view_mut()
                .sub_assign(diff_p * self.alpha);
        }
    }
}

impl<T: Num> Optimizer<T> for Adam<T> {
    fn update(&self, parameters: &[Variable<T>]) {
        if self.m.borrow().is_empty() {
            self.init_m(parameters);
        }
        if self.v.borrow().is_empty() {
            self.init_v(parameters);
        }

        self.update_m(parameters);
        self.update_v(parameters);
        self.update_parameters(parameters);
    }
}

#[cfg(test)]
mod adam {
    use zenu_autograd::{
        creator::from_vec::from_vec,
        functions::{loss::mse::mean_squared_error, matmul::matmul},
        Variable,
    };
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use crate::Optimizer;

    use super::Adam;

    fn simple_function(
        x: Variable<f64>,
        weight1: Variable<f64>,
        weight2: Variable<f64>,
    ) -> Variable<f64> {
        let x = matmul(x, weight1);
        matmul(x, weight2)
    }

    fn adam_apply(
        adam: &Adam<f64>,
        forward_func: fn(Variable<f64>, Variable<f64>, Variable<f64>) -> Variable<f64>,
        input: Variable<f64>,
        target: Variable<f64>,
        weight1: Variable<f64>,
        weight2: Variable<f64>,
    ) {
        let output = forward_func(input.clone(), weight1.clone(), weight2.clone());
        let loss = mean_squared_error(target, output);
        loss.backward();
        adam.update(&[weight1.clone(), weight2.clone()]);
        loss.clear_grad();
    }

    #[test]
    fn small_3_times() {
        let adam = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let weight1 = from_vec(vec![1., 2., 3., 4.], [2, 2]);
        let weight2 = from_vec(vec![1., 2.], [2, 1]);

        let input = from_vec(vec![1., 2.], [1, 2]);
        let target = from_vec(vec![1.], [1, 1]);

        adam_apply(
            &adam,
            simple_function,
            input,
            target,
            weight1.clone(),
            weight2.clone(),
        );

        let weight_1_ans = OwnedMatrixDyn::from_vec(vec![0.9900, 1.9900, 2.99, 3.99], [2, 2]);
        let weight_2_ans = OwnedMatrixDyn::from_vec(vec![0.990, 1.990], [2, 1]);
        assert!((weight1.get_data() - weight_1_ans).asum() < 1e-2);
        assert!((weight2.get_data() - weight_2_ans).asum() < 1e-2);

        let input = from_vec(vec![1., 2.], [1, 2]);
        let target = from_vec(vec![1.], [1, 1]);
        adam_apply(
            &adam,
            simple_function,
            input.clone(),
            target.clone(),
            weight1.clone(),
            weight2.clone(),
        );

        let weight_1_ans = OwnedMatrixDyn::from_vec(vec![0.98, 1.98, 2.98, 3.98], [2, 2]);
        let weight_2_ans = OwnedMatrixDyn::from_vec(vec![0.98, 1.98], [2, 1]);
        assert!((weight1.get_data() - weight_1_ans).asum() < 1e-1);
        assert!((weight2.get_data() - weight_2_ans).asum() < 1e-1);
    }
}

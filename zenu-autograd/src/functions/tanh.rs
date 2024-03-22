use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{num::Num, operation::basic_operations::MatrixTanh};

use crate::{creator::zeros::zeros_like, Function, Variable, VariableWeak};

use super::cosh::cosh;

struct Tanh<T: Num> {
    input: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> Tanh<T> {
    pub fn new(input: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num> Function<T> for Tanh<T> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        output.get_data_mut().tanh(self.input.get_data());
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        let input_cosh = cosh(self.input.clone());
        let input_cosh_2 = input_cosh.clone() * input_cosh.clone();
        let grad = output_grad / input_cosh_2;
        self.input.set_grad(grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.input.clone()]
    }
}

pub fn tanh<T: Num>(input: Variable<T>) -> Variable<T> {
    let output = zeros_like(&input);
    let tanh = Tanh::new(input, output.clone());
    tanh.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(tanh))));
    output
}

#[cfg(test)]
mod tanh {
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use crate::creator::from_vec::from_vec;

    use super::tanh;

    #[test]
    fn tanh_1d() {
        let x = from_vec(vec![1., 2., 3., 4., 5., 6.], [6]);
        let y = tanh(x.clone());
        y.backward();
        let y_data = y.get_data();
        let x_grad = x.get_grad().unwrap().get_data();
        let y_ans = OwnedMatrixDyn::from_vec(
            vec![
                1_f64.tanh(),
                2_f64.tanh(),
                3_f64.tanh(),
                4_f64.tanh(),
                5_f64.tanh(),
                6_f64.tanh(),
            ],
            [6],
        );
        let x_grad_ans = OwnedMatrixDyn::from_vec(
            vec![
                1. / (1_f64.cosh() * 1_f64.cosh()),
                1. / (2_f64.cosh() * 2_f64.cosh()),
                1. / (3_f64.cosh() * 3_f64.cosh()),
                1. / (4_f64.cosh() * 4_f64.cosh()),
                1. / (5_f64.cosh() * 5_f64.cosh()),
                1. / (6_f64.cosh() * 6_f64.cosh()),
            ],
            [6],
        );
        let diff_y = (y_data - y_ans).asum();
        let diff_x_grad = (x_grad.clone() - x_grad_ans.clone()).asum();
        assert!(diff_y < 1e-10);
        assert!(diff_x_grad < 1e-10);
    }
}

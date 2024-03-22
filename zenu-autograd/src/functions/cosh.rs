use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{num::Num, operation::basic_operations::MatrixCosh};

use crate::{creator::zeros::zeros_like, Function, Variable, VariableWeak};

use super::sinh::sinh;

struct CosH<T: Num> {
    input: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> CosH<T> {
    pub fn new(input: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num> Function<T> for CosH<T> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        output.get_data_mut().cosh(self.input.get_data());
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        let input_grad = sinh(self.input.clone()) * output_grad;
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.input.clone()]
    }
}

pub fn cosh<T: Num>(input: Variable<T>) -> Variable<T> {
    let output = zeros_like(&input);
    let cosh = CosH::new(input, output.clone());
    cosh.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(cosh))));
    output
}

#[cfg(test)]
mod cosh {
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use crate::creator::from_vec::from_vec;

    use super::cosh;

    #[test]
    fn cosh_1d() {
        let x = from_vec(vec![1., 2., 3., 4., 5., 6.], [6]);
        let y = cosh(x.clone());
        y.backward();
        let y_data = y.get_data();
        let x_grad = x.get_grad().unwrap().get_data();
        let y_ans = OwnedMatrixDyn::from_vec(
            vec![
                1_f64.cosh(),
                2_f64.cosh(),
                3_f64.cosh(),
                4_f64.cosh(),
                5_f64.cosh(),
                6_f64.cosh(),
            ],
            [6],
        );
        let x_grad_ans = OwnedMatrixDyn::from_vec(
            vec![
                1_f64.sinh(),
                2_f64.sinh(),
                3_f64.sinh(),
                4_f64.sinh(),
                5_f64.sinh(),
                6_f64.sinh(),
            ],
            [6],
        );
        let diff_y = y_data - y_ans;
        let diff_x = x_grad - x_grad_ans;
        let diff_y_asum = diff_y.asum();
        let diff_x_asum = diff_x.asum();
        assert!(diff_y_asum < 1e-7);
        assert!(diff_x_asum < 1e-7);
    }

    #[test]
    fn test_2d() {
        let x = from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [3, 4],
        );
        let y = cosh(x.clone());
        y.backward();
        let y_data = y.get_data();
        let x_grad = x.get_grad().unwrap().get_data();
        let y_ans = OwnedMatrixDyn::from_vec(
            vec![
                1_f64.cosh(),
                2_f64.cosh(),
                3_f64.cosh(),
                4_f64.cosh(),
                5_f64.cosh(),
                6_f64.cosh(),
                7_f64.cosh(),
                8_f64.cosh(),
                9_f64.cosh(),
                10_f64.cosh(),
                11_f64.cosh(),
                12_f64.cosh(),
            ],
            [3, 4],
        );
        let x_grad_ans = OwnedMatrixDyn::from_vec(
            vec![
                1_f64.sinh(),
                2_f64.sinh(),
                3_f64.sinh(),
                4_f64.sinh(),
                5_f64.sinh(),
                6_f64.sinh(),
                7_f64.sinh(),
                8_f64.sinh(),
                9_f64.sinh(),
                10_f64.sinh(),
                11_f64.sinh(),
                12_f64.sinh(),
            ],
            [3, 4],
        );
        let diff_y = y_data - y_ans;
        let diff_x = x_grad - x_grad_ans;
        let diff_y_asum = diff_y.asum();
        let diff_x_asum = diff_x.asum();
        assert!(diff_y_asum < 1e-7);
        assert!(diff_x_asum < 1e-7);
    }
}

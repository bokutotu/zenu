use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::alloc::alloc_like, Function, Variable, VariableWeak};

use super::cosh::cosh;

struct SinH<T: Num, D: Device> {
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> SinH<T, D> {
    pub fn new(input: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for SinH<T, D> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        output
            .get_data_mut()
            .to_ref_mut()
            .sinh_array(&self.input.get_data());
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        let input_grad = cosh(self.input.clone()) * output_grad;
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

#[must_use]
pub fn sinh<T: Num, D: Device>(input: Variable<T, D>) -> Variable<T, D> {
    let output = alloc_like(&input);
    let sinh = SinH::new(input, output.clone());
    sinh.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(sinh))));
    output
}

#[cfg(test)]
mod sinh_test {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::creator::from_vec::from_vec;

    use super::sinh;

    fn sinh_1d<D: Device>() {
        let x = from_vec(vec![1., 2., 3., 4., 5., 6.], [6]);
        let y = sinh(x.clone());
        y.backward();
        let y_ans: Matrix<Owned<f64>, DimDyn, D> = Matrix::from_vec(
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
        let x_ans: Matrix<Owned<f64>, DimDyn, D> = Matrix::from_vec(
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
        assert_val_eq!(y, y_ans, 1e-6);
        assert_val_eq_grad!(x, x_ans, 1e-6);
    }
    run_test!(sinh_1d, sinh_1d_cpu, sinh_1d_nvidia);

    fn sinh_2d<D: Device>() {
        let x = from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [3, 4],
        );
        let y = sinh(x.clone());
        y.backward();
        let y_ans: Matrix<Owned<f64>, DimDyn, D> = Matrix::from_vec(
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
        let x_grad_ans: Matrix<Owned<f64>, DimDyn, D> = Matrix::from_vec(
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
        assert_val_eq!(y, y_ans, 1e-6);
        assert_val_eq_grad!(x, x_grad_ans, 1e-6);
    }
    run_test!(sinh_2d, sinh_2d_cpu, sinh_2d_nvidia);
}

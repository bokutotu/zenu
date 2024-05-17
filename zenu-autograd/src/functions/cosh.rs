use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::zeros::zeros_like, Function, Variable, VariableWeak};

use super::sinh::sinh;

struct CosH<T: Num, D: Device> {
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> CosH<T, D> {
    pub fn new(input: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for CosH<T, D> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        output
            .get_data_mut()
            .to_ref_mut()
            .cosh_array(&self.input.get_data());
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        let input_grad = sinh(self.input.clone()) * output_grad;
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

pub fn cosh<T: Num, D: Device>(input: Variable<T, D>) -> Variable<T, D> {
    let output = zeros_like(&input);
    let cosh = CosH::new(input, output.clone());
    cosh.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(cosh))));
    output
}

#[cfg(test)]
mod cosh {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::creator::from_vec::from_vec;

    use super::cosh;

    fn cosh_1d<D: Device>() {
        let x = from_vec(vec![1., 2., 3., 4., 5., 6.], [6]);
        let y = cosh(x.clone());
        y.backward();
        let y_ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![
                1_f32.cosh(),
                2_f32.cosh(),
                3_f32.cosh(),
                4_f32.cosh(),
                5_f32.cosh(),
                6_f32.cosh(),
            ],
            [6],
        );
        let x_grad_ans: Matrix<_, DimDyn, D> = Matrix::from_vec(
            vec![
                1_f32.sinh(),
                2_f32.sinh(),
                3_f32.sinh(),
                4_f32.sinh(),
                5_f32.sinh(),
                6_f32.sinh(),
            ],
            [6],
        );
        assert_val_eq!(y, y_ans, 1e-7);
        assert_val_eq_grad!(x, x_grad_ans, 1e-7);
    }
    run_test!(cosh_1d, cosh_1d_cpu, cosh_1d_nvidia);

    fn test_2d<D: Device>() {
        let x = from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [3, 4],
        );
        let y = cosh(x.clone());
        y.backward();
        let y_ans: Matrix<_, DimDyn, D> = Matrix::from_vec(
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
        let x_grad_ans: Matrix<_, DimDyn, D> = Matrix::from_vec(
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
        assert_val_eq!(y, y_ans, 1e-7);
        assert_val_eq_grad!(x, x_grad_ans, 1e-7);
    }
    run_test!(test_2d, test_2d_cpu, test_2d_nvidia);
}

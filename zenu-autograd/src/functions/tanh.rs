use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::alloc::alloc_like, Function, Variable, VariableWeak};

use super::cosh::cosh;

struct Tanh<T: Num, D: Device> {
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Tanh<T, D> {
    pub fn new(input: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for Tanh<T, D> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        output
            .get_data_mut()
            .to_ref_mut()
            .tanh_array(&self.input.get_data().to_ref());
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        let input_cosh = cosh(self.input.clone());
        let input_cosh_2 = input_cosh.clone() * input_cosh.clone();
        let grad = output_grad / input_cosh_2;
        self.input.set_grad(grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

pub fn tanh<T: Num, D: Device>(input: Variable<T, D>) -> Variable<T, D> {
    let output = alloc_like(&input);
    let tanh = Tanh::new(input, output.clone());
    tanh.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(tanh))));
    output
}

#[cfg(test)]
mod tanh {

    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::creator::from_vec::from_vec;

    use super::tanh;

    fn tanh_1d<D: Device>() {
        let x = from_vec(vec![1f32, 2., 3., 4., 5., 6.], [6]);
        let y = tanh(x.clone());
        y.backward();
        let y_ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![
                1_f32.tanh(),
                2_f32.tanh(),
                3_f32.tanh(),
                4_f32.tanh(),
                5_f32.tanh(),
                6_f32.tanh(),
            ],
            [6],
        );
        let x_grad_ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![
                1. / (1_f32.cosh() * 1_f32.cosh()),
                1. / (2_f32.cosh() * 2_f32.cosh()),
                1. / (3_f32.cosh() * 3_f32.cosh()),
                1. / (4_f32.cosh() * 4_f32.cosh()),
                1. / (5_f32.cosh() * 5_f32.cosh()),
                1. / (6_f32.cosh() * 6_f32.cosh()),
            ],
            [6],
        );
        assert_val_eq!(y, y_ans, 1e-6);
        assert_val_eq_grad!(x, x_grad_ans, 1e-6);
    }
    run_test!(tanh_1d, tanh_1d_cpu, tanh_1d_nvidia);
}

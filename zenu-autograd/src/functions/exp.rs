use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::alloc::alloc_like, Function, Variable, VariableWeak};

struct Exp<T: Num, D: Device> {
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Exp<T, D> {
    pub fn new(input: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for Exp<T, D> {
    fn forward(&self) {
        let input = self.input.get_data();
        let output = self.output.upgrade().unwrap();
        output.get_data_mut().to_ref_mut().exp_array(&input);
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        self.input.set_grad(output * output_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

pub fn exp<T: Num, D: Device>(input: Variable<T, D>) -> Variable<T, D> {
    let output = alloc_like(&input);
    let exp = Exp::new(input, output.clone());
    exp.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(exp))));
    output
}

#[cfg(test)]
mod exp {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::creator::from_vec::from_vec;

    use super::exp;

    fn exp_1d<D: Device>() {
        let x = from_vec(vec![1., 2., 3.], [3]);
        let exp = exp(x.clone());
        exp.backward();
        let exp_ans: Matrix<Owned<f64>, DimDyn, D> =
            Matrix::from_vec(vec![2.7182817, 7.389056, 20.085537], [3]);
        let x_grad: Matrix<Owned<f64>, DimDyn, D> =
            Matrix::from_vec(vec![1_f64.exp(), 2_f64.exp(), 3_f64.exp()], [3]);
        assert_val_eq!(exp, exp_ans, 1e-6);
        assert_val_eq_grad!(x, x_grad, 1e-6);
    }
    run_test!(exp_1d, exp_1d_cpu, exp_1d_nvidia);
}

use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{matrix::ToViewMutMatrix, num::Num, operation::exp::ExpAssign};

use crate::{creator::zeros::zeros_like, Function, Variable, VariableWeak};

struct Exp<T: Num> {
    input: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> Exp<T> {
    pub fn new(input: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num> Function<T> for Exp<T> {
    fn forward(&self) {
        let input = self.input.get_data();
        let output = self.output.upgrade().unwrap();
        output.get_data_mut().to_view_mut().exp_assign(&input);
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        self.input.set_grad(output * output_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.input.clone()]
    }
}

pub fn exp<T: Num>(input: Variable<T>) -> Variable<T> {
    let output = zeros_like(&input);
    let exp = Exp::new(input, output.clone());
    exp.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(exp))));
    output
}

#[cfg(test)]
mod exp {
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use crate::creator::from_vec::from_vec;

    use super::exp;

    #[test]
    fn exp_1d() {
        let x = from_vec(vec![1., 2., 3.], [3]);
        let exp = exp(x);
        exp.backward();
        let exp_data = exp.get_data();
        let exp_grad_data = exp.get_grad().unwrap().get_data();
        let exp_ans = OwnedMatrixDyn::from_vec(vec![2.7182817, 7.389056, 20.085537], [3]);
        let exp_grad_ans = OwnedMatrixDyn::from_vec(vec![1., 1., 1.], [3]);
        let forward_diff = exp_data - exp_ans;
        let grad_diff = exp_grad_data - exp_grad_ans;
        let forward_diff_asum = forward_diff.asum();
        let grad_diff_asum = grad_diff.asum();
        assert!(forward_diff_asum < 1e-6);
        assert!(grad_diff_asum < 1e-6);
    }
}

use std::{
    cell::RefCell,
    ops::{DerefMut, Mul},
    rc::Rc,
};

use zenu_matrix::{
    constructor::zeros::Zeros,
    matrix::{MatrixBase, ToViewMatrix},
    num::Num,
    operation::basic_operations::MatrixMul,
};

use crate::{Function, Variable, VariableWeak};

use super::{output_shape, sum_to::sum_to};

struct Multiply<T: Num> {
    x: Variable<T>,
    y: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> Multiply<T> {
    pub fn new(x: Variable<T>, y: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<T: Num> Function<T> for Multiply<T> {
    fn forward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let x = x.to_view();
        let y = y.to_view();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        MatrixMul::mul(output.deref_mut(), x, y);
    }

    fn backward(&self) {
        let x_shape = self.x.get_data().shape();
        let y_shape = self.y.get_data().shape();
        let output = self.output.upgrade().unwrap();
        let grad = output.get_grad().clone().unwrap();
        let x_grad = grad.clone() * self.y.clone();
        let y_grad = self.x.clone() * grad;
        self.x.set_grad(sum_to(x_grad, x_shape));
        self.y.set_grad(sum_to(y_grad, y_shape));
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

fn mul<T: Num>(x: Variable<T>, y: Variable<T>) -> Variable<T> {
    let output_shape = output_shape(&x, &y);
    let output = Zeros::zeros(output_shape);
    let output = Variable::new(output);
    let mul = Multiply::new(x, y, output.clone());
    mul.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(mul))));
    output
}

impl<T: Num> Mul<Variable<T>> for Variable<T> {
    type Output = Variable<T>;

    fn mul(self, rhs: Variable<T>) -> Self::Output {
        mul(self, rhs)
    }
}

#[cfg(test)]
mod mul {
    use zenu_matrix::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::asum::Asum,
    };

    use crate::Variable;

    #[test]
    fn mul_2d_1d() {
        let a_mat: OwnedMatrixDyn<f32> =
            OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let b_mat: OwnedMatrixDyn<f32> = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], [3]);
        let a = Variable::new(a_mat);
        let b = Variable::new(b_mat);
        let c = a.clone() * b.clone();
        let c_ans = OwnedMatrixDyn::from_vec(vec![1., 4., 9., 4., 10., 18.], [2, 3]);
        let diff = c.get_data().to_view() - c_ans.to_view();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);
        c.backward();

        a.with_grad_data(|grad| {
            let grad = grad.to_view();
            let ans = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 1., 2., 3.], [2, 3]);
            let diff = grad - ans.to_view();
            let diff_sum = diff.asum();
            assert!(diff_sum < 1e-6);
        });
        b.with_grad_data(|grad| {
            let grad = grad.to_view();
            let ans = OwnedMatrixDyn::from_vec(vec![5., 7., 9.], [3]);
            let diff = grad - ans.to_view();
            let diff_sum = diff.asum();
            assert!(diff_sum < 1e-6);
        });
    }
}

use std::{cell::RefCell, ops::Mul, rc::Rc};

use ruml_matrix::{
    dim::DimDyn,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory_impl::OwnedMem,
    num::Num,
    operation::{mul::MatrixMul, ones::Ones, zeros::Zeros},
};

use crate::{Function, Variable, VariableWeak};

use super::{gradient_sum_over_axis, output_shape};

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
        let output = output.to_view_mut();
        MatrixMul::mul(output, x, y);
    }

    fn backward(&self) {
        let x_shape = self.x.get_data().shape();
        let y_shape = self.y.get_data().shape();
        let mut x_grad_: Matrix<OwnedMem<T>, DimDyn> = Ones::ones(x_shape);
        let mut y_grad_: Matrix<OwnedMem<T>, DimDyn> = Ones::ones(y_shape);
        self.output.upgrade().unwrap().with_grad_data(|grad| {
            let x = self.x.get_data();
            let y = self.y.get_data();
            let grad = grad.to_view();
            let x_grad = grad.clone() * y.to_view();
            let y_grad = grad * x.to_view();
            gradient_sum_over_axis(x_grad.to_view(), x_grad_.to_view_mut());
            gradient_sum_over_axis(y_grad.to_view(), y_grad_.to_view_mut());
        });
        *self.x.get_grad_mut() = Some(Variable::new(x_grad_));
        *self.y.get_grad_mut() = Some(Variable::new(y_grad_));
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
    use ruml_matrix::{
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
            let diff = dbg!(grad - ans.to_view());
            let diff_sum = diff.asum();
            assert!(diff_sum < 1e-6);
        });
        b.with_grad_data(|grad| {
            let grad = grad.to_view();
            let ans = OwnedMatrixDyn::from_vec(vec![5., 7., 9.], [3]);
            let diff = dbg!(grad - ans.to_view());
            let diff_sum = diff.asum();
            assert!(diff_sum < 1e-6);
        });
    }
}

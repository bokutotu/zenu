use std::{cell::RefCell, ops::Mul, rc::Rc};

use ruml_matrix::{
    dim::DimDyn,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::OwnedMemory,
    operation::{mul::MatrixMul, zeros::Zeros},
};

use crate::{Function, Variable, VariableWeak};

use super::gradient_sum_over_axis;

struct Multiply<M: OwnedMemory> {
    x: Variable<M>,
    y: Variable<M>,
    output: VariableWeak<M>,
}

impl<M: OwnedMemory> Multiply<M> {
    pub fn new(x: Variable<M>, y: Variable<M>, output: Variable<M>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<M: OwnedMemory> Function<M> for Multiply<M> {
    fn forward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        MatrixMul::mul(output.to_view_mut(), x.to_view(), y.to_view());
    }

    fn backward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let x_shape = x.shape();
        let y_shape = y.shape();
        let mut x_grad_: Matrix<M, DimDyn> = Zeros::zeros(x_shape);
        let mut y_grad_: Matrix<M, DimDyn> = Zeros::zeros(y_shape);
        self.output.upgrade().unwrap().with_grad_data(|grad| {
            let grad = grad.to_view();
            let x_grad = grad.clone() * y.to_view();
            let y_grad = grad * x.to_view();
            gradient_sum_over_axis(x_grad.to_view(), x_grad_.to_view_mut());
            gradient_sum_over_axis(y_grad.to_view(), y_grad_.to_view_mut());
        });
    }

    fn get_inputs(&self) -> Vec<Variable<M>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

fn mul<M: OwnedMemory>(x: Variable<M>, y: Variable<M>) -> Variable<M> {
    let output_shape = if x.get_data().shape().is_include(&y.get_data().shape()) {
        x.get_data().shape()
    } else {
        y.get_data().shape()
    };
    let output = Zeros::zeros(output_shape);
    let output = Variable::new(output);
    let mul = Multiply::new(x, y, output.clone());
    mul.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(mul))));
    output
}

impl<M: OwnedMemory> Mul<Variable<M>> for Variable<M> {
    type Output = Variable<M>;

    fn mul(self, rhs: Variable<M>) -> Self::Output {
        mul(self, rhs)
    }
}

#[cfg(test)]
mod mul {
    use ruml_matrix::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::CpuOwnedMatrixDyn,
        operation::asum::Asum,
    };

    use crate::Variable;

    #[test]
    fn mul_2d_1d() {
        let a_mat: CpuOwnedMatrixDyn<f32> =
            CpuOwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let b_mat: CpuOwnedMatrixDyn<f32> = CpuOwnedMatrixDyn::from_vec(vec![1., 2., 3.], [3]);
        let a = Variable::new(a_mat);
        let b = Variable::new(b_mat);
        let c = a.clone() * b.clone();
        let c_ans = CpuOwnedMatrixDyn::from_vec(vec![1., 4., 9., 4., 10., 18.], [2, 3]);
        let diff = c.get_data().to_view() - c_ans.to_view();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);
        c.backward();

        a.with_grad_data(|grad| {
            let grad = grad.to_view();
            let ans = CpuOwnedMatrixDyn::from_vec(vec![1., 2., 3., 1., 2., 3.], [2, 3]);
            let diff = dbg!(grad - ans.to_view());
            let diff_sum = diff.asum();
            assert!(diff_sum < 1e-6);
        });
        b.with_grad_data(|grad| {
            let grad = grad.to_view();
            let ans = CpuOwnedMatrixDyn::from_vec(vec![5., 7., 9.], [3]);
            let diff = dbg!(grad - ans.to_view());
            let diff_sum = diff.asum();
            assert!(diff_sum < 1e-6);
        });
    }
}

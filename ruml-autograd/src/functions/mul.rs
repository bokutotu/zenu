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
        let x_shape = self.x.get_data().shape();
        let y_shape = self.y.get_data().shape();
        let mut x_grad_: Matrix<M, DimDyn> = Zeros::zeros(x_shape);
        let mut y_grad_: Matrix<M, DimDyn> = Zeros::zeros(y_shape);
        self.output.upgrade().unwrap().with_grad_data(|grad| {
            let grad = grad.to_view();
            let x = self.x.get_data();
            let y = self.y.get_data();
            let x_grad = grad.clone() * x.to_view();
            let y_grad = grad * y.to_view();
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
    output.set_creator(Rc::new(RefCell::new(Box::new(mul))));
    output
}

impl<M: OwnedMemory> Mul<Variable<M>> for Variable<M> {
    type Output = Variable<M>;

    fn mul(self, rhs: Variable<M>) -> Self::Output {
        mul(self, rhs)
    }
}

use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    dim::LessDimTrait,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::OwnedMatrixDyn,
    num::Num,
    operation::{copy_from::CopyFrom, sum::MatrixSum, zeros::Zeros},
};

use crate::{Function, Variable, VariableWeak};

use super::broadcast::broadcast;

struct Sum<T: Num> {
    input: Variable<T>,
    output: VariableWeak<T>,
    axis: usize,
    keep_dim: bool,
}

impl<T: Num> Sum<T> {
    pub fn new(input: Variable<T>, output: VariableWeak<T>, axis: usize, keep_dim: bool) -> Self {
        Self {
            input,
            output,
            axis,
            keep_dim,
        }
    }
}

impl<T: Num> Function<T> for Sum<T> {
    fn forward(&self) {
        let input = self.input.get_data();
        let input = input.to_view();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        let mut output = output.to_view_mut();
        let ans = input.sum(self.axis, self.keep_dim);
        output.copy_from(&ans.to_view());
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().clone().unwrap();
        let input_grad = broadcast(output_grad, self.input.get_data().shape());
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.input.clone()]
    }
}

pub fn sum<T: Num>(input: Variable<T>, axis: usize, keep_dim: bool) -> Variable<T> {
    let output_shape = input.get_data().shape().remove_axis(axis);
    let output = Variable::from(OwnedMatrixDyn::zeros(output_shape));
    let sum = Sum::new(input, output.clone().downgrade(), axis, keep_dim);
    sum.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(sum))));
    output
}

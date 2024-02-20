use std::{cell::RefCell, ops::Add, rc::Rc};

use ruml_matrix::{
    dim::DimDyn,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::OwnedMemory,
    operation::{add::MatrixAdd, zeros::Zeros},
};

use crate::{Function, Variable, VariableWeak};

use super::gradient_sum_over_axis;

pub(crate) struct Addition<M: OwnedMemory> {
    x: Variable<M>,
    y: Variable<M>,
    output: VariableWeak<M>,
}

impl<M: OwnedMemory> Addition<M> {
    pub fn new(x: Variable<M>, y: Variable<M>, output: Variable<M>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<M: OwnedMemory> Function<M> for Addition<M> {
    fn forward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        MatrixAdd::add(output.to_view_mut(), x.to_view(), y.to_view());
    }

    fn backward(&self) {
        let x_shape = self.x.get_data().shape();
        let y_shape = self.y.get_data().shape();
        let mut x_grad: Matrix<M, DimDyn> = Zeros::zeros(x_shape);
        let mut y_grad: Matrix<M, DimDyn> = Zeros::zeros(y_shape);
        self.output.upgrade().unwrap().with_grad_data(|grad| {
            gradient_sum_over_axis(grad.to_view(), x_grad.to_view_mut());
            gradient_sum_over_axis(grad.to_view(), y_grad.to_view_mut());
        });
    }

    fn get_inputs(&self) -> Vec<Variable<M>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

pub(crate) fn add<M: OwnedMemory>(x: Variable<M>, y: Variable<M>) -> Variable<M> {
    let output_shape: DimDyn = if x.get_data().shape().is_include(&y.get_data().shape()) {
        x.get_data().shape()
    } else {
        y.get_data().shape()
    };
    let output = Zeros::zeros(output_shape);
    let output = Variable::new(output);
    let add = Addition::new(x, y, output.clone());
    add.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(add))));
    output
}

impl<M: OwnedMemory> Add<Variable<M>> for Variable<M> {
    type Output = Variable<M>;

    fn add(self, other: Variable<M>) -> Self::Output {
        add(self, other)
    }
}

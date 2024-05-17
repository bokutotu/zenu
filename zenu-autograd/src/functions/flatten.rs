use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    dim::DimTrait,
    matrix::MatrixBase,
    num::Num,
    operation::{copy_from::CopyFrom, reshape::Reshape},
};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

use super::reshape::reshape;

struct Flatten<T: Num> {
    input: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num, D: Device> Flatten<T> {
    fn new(input: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num, D: Device> Function<T> for Flatten<T> {
    fn forward(&self) {
        let output_shape = self.output.upgrade().unwrap().get_data().shape();
        let input_mat = self.input.get_data();
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .copy_from(&input_mat.reshape(output_shape.slice()));
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().unwrap();
        self.input
            .set_grad(reshape(output_grad, self.input.get_data().shape().slice()));
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.input.clone()]
    }
}

pub fn flatten<T: Num>(input: Variable<T>) -> Variable<T> {
    let input_shape = input.get_data().shape();
    let batch_size = input_shape[0];
    let num_elm = input_shape.num_elm();
    let output_shape = [batch_size, num_elm / batch_size];
    let output = zeros(output_shape);
    let flatten = Flatten::new(input, output.clone());
    flatten.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(flatten))));
    output
}

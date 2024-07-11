use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, dim::DimTrait, num::Num};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

use super::reshape::reshape;

struct Flatten<T: Num, D: Device> {
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Flatten<T, D> {
    fn new(input: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for Flatten<T, D> {
    fn forward(&self) {
        let output_shape = self.output.upgrade().unwrap().get_data().shape();
        let input_mat = self.input.get_data();
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&input_mat.reshape(output_shape.slice()));
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().unwrap();
        let input_shape = self.input.get_shape();
        let input_grad = reshape(output_grad, input_shape.slice());
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

pub fn flatten<T: Num, D: Device>(input: Variable<T, D>) -> Variable<T, D> {
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

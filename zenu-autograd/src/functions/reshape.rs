use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    dim::DimTrait,
    matrix::{MatrixBase, ToViewMutMatrix},
    num::Num,
    operation::{copy_from::CopyFrom, reshape::Reshape as MatrixReshape},
};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

struct Reshape<T: Num, D: Device> {
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Reshape<T, D> {
    fn new(input: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for Reshape<T, D> {
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
        self.input.set_grad(reshape(
            output_grad.clone(),
            self.input.get_data().shape().slice(),
        ));
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

pub fn reshape<T: Num, D: Device>(input: Variable<T, D>, output_shape: &[usize]) -> Variable<T, D> {
    let output = zeros(output_shape);
    let reshape = Reshape::new(input, output.clone());
    reshape.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(reshape))));
    output
}

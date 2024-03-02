use std::{cell::RefCell, rc::Rc};

use ruml_matrix::{
    dim::DimDyn, matrix::MatrixBase, matrix_impl::Matrix, memory_impl::OwnedMem, num::Num,
    operation::zeros::Zeros,
};

use crate::{Function, Variable, VariableWeak};

struct Relu<T: Num> {
    input: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> Relu<T> {
    pub fn new(input: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num> Function<T> for Relu<T> {
    fn forward(&self) {
        todo!();
    }

    fn backward(&self) {
        todo!();
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.input.clone()]
    }
}

pub fn relu<T: Num>(input: Variable<T>) -> Variable<T> {
    let output: Matrix<OwnedMem<T>, DimDyn> = Zeros::zeros(input.get_data().shape());
    let output = Variable::from(output);
    let relu = Relu::new(input, output.clone());
    output.set_creator(Rc::new(RefCell::new(Box::new(relu))));
    output
}

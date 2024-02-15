use ruml_matrix::{matrix::OwnedMatrix, num::Num};

use crate::{Function, Variable, VariableWeak};

pub struct Add<T: OwnedMatrix> {
    x: Variable<T>,
    y: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: OwnedMatrix> Add<T> {
    pub fn new(x: Variable<T>, y: Variable<T>, output: VariableWeak<T>) -> Self {
        Self { x, y, output }
    }
}

impl<T: OwnedMatrix> Function<T> for Add<T> {
    fn forward(&self) {
        todo!();
    }

    fn backward(&self) {
        todo!();
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        // vec![self.x.clone(), self.y.clone()]
        todo!();
    }
}

use zenu_matrix::num::Num;

use crate::{Function, Variable, VariableWeak};

pub struct CrossEntropy<T: Num> {
    x: Variable<T>,
    y: Variable<T>,
    loss: VariableWeak<T>,
}

impl<T: Num> CrossEntropy<T> {
    pub fn new(x: Variable<T>, y: Variable<T>, loss: VariableWeak<T>) -> Self {
        CrossEntropy { x, y, loss }
    }
}

impl<T: Num> Function<T> for CrossEntropy<T> {
    fn forward(&self) {
        todo!()
    }

    fn backward(&self) {
        todo!()
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

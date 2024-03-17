use zenu_autograd::{Function, Variable};
use zenu_matrix::num::Num;

use crate::Layer;

pub struct Linear<T: Num> {
    weight: Variable<T>,
    bias: Variable<T>,
}

impl<T: Num> Function<T> for Linear<T> {
    fn forward(&self) {
        unimplemented!()
    }

    fn backward(&self) {
        unimplemented!()
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        unimplemented!()
    }
}

impl<T: Num> Layer<T> for Linear<T> {
    fn init_parameters(&self) {
        unimplemented!()
    }

    fn parameters(&self) -> Vec<Variable<T>> {
        unimplemented!()
    }
}

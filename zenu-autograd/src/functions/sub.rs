use std::ops::Sub;

use zenu_matrix::{device::Device, num::Num};

use crate::{Function, Variable};

pub struct SubFunc<T: Num, D: Device> {
    x: Variable<T, D>,
    y: Variable<T, D>,
    output: Variable<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for SubFunc<T, D> {
    fn forward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let output = x.to_ref() - y.to_ref();
        self.output
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&output.to_ref());
    }

    fn backward(&self) {
        let output_grad = self.output.get_grad().clone().unwrap();
        let x_grad = output_grad.clone();
        let y_grad = output_grad.clone() * Variable::from(T::minus_one());
        self.x.set_grad(x_grad);
        self.y.set_grad(y_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

pub fn sub<T: Num, D: Device>(x: Variable<T, D>, y: Variable<T, D>) -> Variable<T, D> {
    let y = y * Variable::from(T::minus_one());
    x + y
}

impl<T: Num, D: Device> Sub<Variable<T, D>> for Variable<T, D> {
    type Output = Variable<T, D>;

    fn sub(self, rhs: Variable<T, D>) -> Self::Output {
        sub(self, rhs)
    }
}

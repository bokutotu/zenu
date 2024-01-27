use std::{cell::RefCell, rc::Rc};

use crate::{
    variable::{Variable, VariableWeak},
    Value,
};

use super::Function;

pub struct Add<V> {
    x: Variable<V>,
    y: Variable<V>,
    output: VariableWeak<V>,
}

impl<V: Value> Add<V> {
    pub fn new(x: Variable<V>, y: Variable<V>, output: Variable<V>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<V: Value> Function<V> for Add<V> {
    fn forward(&self) {
        let ans = self.x.get_data() + self.y.get_data();
        self.output.upgrade().unwrap().set_data(ans);
    }

    fn backward(&self) {
        let input_grad = self.output.upgrade().unwrap().get_grad().unwrap();
        self.x.set_grad(input_grad.clone());
        self.y.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<V>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

pub fn add<V1: Value, V: AsRef<Variable<V1>>>(x: V, y: V) -> Variable<V1> {
    let x = x.as_ref().clone();
    let y = y.as_ref().clone();
    let output = Variable::new(V1::zero());
    let add = Add::new(x, y, output.clone());
    add.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(add))));
    output
}

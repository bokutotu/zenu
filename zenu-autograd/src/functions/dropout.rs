use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::DimDyn,
    nn::dropout::{dropout as forward, dropout_grad as grad, DropoutState},
    num::Num,
};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

pub struct DropoutConfig<T: Num, D: Device> {
    inner: Rc<RefCell<DropoutState<T, D>>>,
}

impl<T: Num, D: Device> DropoutConfig<T, D> {
    pub fn new(rate: f32) -> Self {
        let inner = Rc::new(RefCell::new(DropoutState::new(rate)));
        Self { inner }
    }

    pub fn gpu_init(&self, shape: DimDyn) {
        self.inner.borrow_mut().gpu_init(shape);
    }
}

struct DropoutForward<T: Num, D: Device> {
    config: DropoutConfig<T, D>,
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
}

struct DropoutBackward<T: Num, D: Device> {
    config: DropoutConfig<T, D>,
    output_grad: Variable<T, D>,
    input_grad: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for DropoutForward<T, D> {
    fn forward(&self) {
        let input = self.input.get_data();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        let mut config = self.config.inner.borrow_mut();
        output.to_ref_mut().copy_from(&forward(&input, &mut config));
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        let input_grad = dropout_backward(output_grad, self.config.inner.borrow().rate);
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

impl<T: Num, D: Device> Function<T, D> for DropoutBackward<T, D> {
    fn forward(&self) {
        let output_grad = self.output_grad.get_data();
        let input_grad = self.input_grad.upgrade().unwrap();
        let mut input_grad = input_grad.get_data_mut();
        let config = self.config.inner.borrow();
        input_grad
            .to_ref_mut()
            .copy_from(&grad(&output_grad, &config));
    }

    fn backward(&self) {
        todo!();
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.output_grad.clone()]
    }
}

pub fn dropout<T: Num, D: Device>(input: Variable<T, D>, rate: f32) -> Variable<T, D> {
    let output = alloc(input.get_shape());

    let config = DropoutConfig::new(rate);

    let dropout = DropoutForward {
        config,
        input,
        output: output.clone().downgrade(),
    };

    dropout.forward();

    output.set_creator(Rc::new(RefCell::new(Box::new(dropout))));

    output.set_name("dropout");

    output
}

fn dropout_backward<T: Num, D: Device>(output_grad: Variable<T, D>, rate: f32) -> Variable<T, D> {
    let input_grad = alloc(output_grad.get_shape());

    let config = DropoutConfig::new(rate);

    let dropout = DropoutBackward {
        config,
        output_grad,
        input_grad: input_grad.clone().downgrade(),
    };

    dropout.forward();

    input_grad.set_creator(Rc::new(RefCell::new(Box::new(dropout))));

    input_grad.set_name("dropout_grad");

    input_grad
}

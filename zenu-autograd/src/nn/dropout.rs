use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::DimDyn,
    nn::dropout::{dropout as forward, dropout_grad as grad, DropoutState},
    num::Num,
};

use crate::{creator::alloc::alloc, is_train, Function, Variable, VariableWeak};

#[derive(Clone)]
pub struct DropoutConfig<T: Num, D: Device> {
    inner: Rc<RefCell<DropoutState<T, D>>>,
}

impl<T: Num, D: Device> DropoutConfig<T, D> {
    #[must_use]
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
        let input_grad = dropout_backward(output_grad, self.config.clone());
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

#[must_use]
pub fn dropout<T: Num, D: Device>(
    input: Variable<T, D>,
    rate: f32,
    config: Option<DropoutConfig<T, D>>,
) -> Variable<T, D> {
    if !is_train() {
        return input;
    }
    let output = alloc(input.get_shape());

    let config = match config {
        Some(config) => config,
        None => DropoutConfig::new(rate),
    };

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

fn dropout_backward<T: Num, D: Device>(
    output_grad: Variable<T, D>,
    config: DropoutConfig<T, D>,
) -> Variable<T, D> {
    let input_grad = alloc(output_grad.get_shape());

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

#[cfg(test)]
mod dropout {
    use zenu_matrix::device::{cpu::Cpu, Device};
    use zenu_test::run_test;

    use crate::creator::rand::normal;

    use super::dropout;

    #[expect(clippy::float_cmp)]
    fn dropout_4d_train<D: Device>() {
        let input = normal::<f32, _, D>(1f32, 1f32, None, [3, 3, 3, 3]);
        let output = dropout(input.clone(), 0.8, None);
        output.backward();

        let input_mat_cpu = {
            let input = input.get_data().clone();
            input.to::<Cpu>()
        };
        let output_mat_cpu = {
            let output = output.get_data().clone();
            output.to::<Cpu>()
        };

        let mask = {
            let s = output_mat_cpu.as_slice();
            s.iter().map(|&x| (x != 0f32)).collect::<Vec<_>>()
        };

        let output_slice = output_mat_cpu.as_slice();
        let input_slice = input_mat_cpu.as_slice();

        for idx in 0..output_slice.len() {
            if mask[idx] {
                let diff = output_slice[idx] - input_slice[idx] / 0.2;
                assert!(
                    diff.abs() < 1e-5,
                    "idx : {} output : {} input slice: {} diff :{}",
                    idx,
                    output_slice[idx],
                    input_slice[idx],
                    diff
                );
            } else {
                assert_eq!(output_slice[idx], 0f32);
            }
        }

        let input_grad = input.get_grad().unwrap();
        let input_grad_cpu = {
            let input_grad = input_grad.get_data().clone();
            input_grad.to::<Cpu>()
        };

        for (idx, mask) in mask.iter().enumerate().take(output_slice.len()) {
            if *mask {
                assert_eq!(input_grad_cpu.as_slice()[idx], 1f32 / (1f32 - 0.8));
            } else {
                assert_eq!(input_grad_cpu.as_slice()[idx], 0f32);
            }
        }
    }
    run_test!(dropout_4d_train, dropout_4d_train_cpu, dropout_4d_train_gpu);
}

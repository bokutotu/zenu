use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    nn::conv::{conv2d_bias_add, conv2d_bias_bkwd},
    num::Num,
};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

use super::conv_without_bias::{conv as conv_without_bias, ConvConfigs};

struct ConvBias<T: Num, D: Device> {
    input: Variable<T, D>,
    bias: Variable<T, D>,
    output: VariableWeak<T, D>,
}

struct ConvBiasGrad<T: Num, D: Device> {
    d_output: Variable<T, D>,
    d_bias: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for ConvBias<T, D> {
    fn forward(&self) {
        conv2d_bias_add(
            self.input.get_as_ref(),
            self.bias.get_as_ref(),
            self.output.upgrade().unwrap().get_as_mut(),
        );
    }

    fn backward(&self) {
        let d_output = self.output.upgrade().unwrap().get_grad().unwrap();
        self.input.set_grad(d_output.clone());
        let d_bias = conv_bias_grad(d_output);
        self.bias.set_grad(d_bias);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone(), self.bias.clone()]
    }
}

impl<T: Num, D: Device> Function<T, D> for ConvBiasGrad<T, D> {
    fn forward(&self) {
        conv2d_bias_bkwd(
            self.d_output.get_as_ref(),
            self.d_bias.upgrade().unwrap().get_as_mut(),
        );
    }

    fn backward(&self) {
        todo!();
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.d_output.clone()]
    }
}

fn conv_bias<T: Num, D: Device>(input: Variable<T, D>, bias: Variable<T, D>) -> Variable<T, D> {
    let output = alloc(input.get_shape());
    let forward = ConvBias {
        input,
        bias,
        output: output.clone().downgrade(),
    };
    forward.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(forward))));
    output
}

fn conv_bias_grad<T: Num, D: Device>(d_output: Variable<T, D>) -> Variable<T, D> {
    let d_bias = alloc(d_output.get_shape());
    let backward = ConvBiasGrad {
        d_output,
        d_bias: d_bias.clone().downgrade(),
    };
    backward.forward();
    d_bias.set_creator(Rc::new(RefCell::new(Box::new(backward))));
    d_bias
}

#[allow(clippy::needless_pass_by_value)]
pub fn conv<T: Num, D: Device>(
    input: Variable<T, D>,
    kernel: Variable<T, D>,
    bias: Option<Variable<T, D>>,
    config: ConvConfigs<T>,
) -> Variable<T, D> {
    let output = conv_without_bias(input.clone(), kernel, config.clone());
    match bias {
        Some(bias) => conv_bias(output, bias),
        None => output,
    }
}

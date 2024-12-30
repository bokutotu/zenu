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
    let d_bias_shape = [1, d_output.get_shape()[1], 1, 1];
    let d_bias = alloc(d_bias_shape);
    let backward = ConvBiasGrad {
        d_output,
        d_bias: d_bias.clone().downgrade(),
    };
    backward.forward();
    d_bias.set_creator(Rc::new(RefCell::new(Box::new(backward))));
    d_bias
}

#[allow(clippy::needless_pass_by_value)]
#[must_use]
pub fn conv<T: Num, D: Device>(
    input: Variable<T, D>,
    filter: Variable<T, D>,
    bias: Option<Variable<T, D>>,
    config: ConvConfigs<T>,
) -> Variable<T, D> {
    let output = conv_without_bias(input.clone(), filter, config.clone());
    match bias {
        Some(bias) => conv_bias(output, bias),
        None => output,
    }
}

#[cfg(test)]
mod conv_bias {
    use std::collections::HashMap;

    use zenu_matrix::{
        device::{cpu::Cpu, Device},
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, read_test_case_from_json_val, run_test};

    use super::conv;
    use crate::{nn::conv::ConvConfigs, Variable};

    fn conv_bias<D: Device>() {
        let map: HashMap<String, Matrix<Owned<f32>, DimDyn, Cpu>> =
            read_test_case_from_json_val!("../test_data_json/conv_bias.json");
        let map: HashMap<String, Matrix<Owned<f32>, DimDyn, D>> = map
            .into_iter()
            .map(|(key, value)| (key, value.to()))
            .collect();
        let x = map.get("input").unwrap().clone();
        let w = map.get("filter").unwrap().clone();
        let b = map.get("bias").unwrap().clone();
        let y = map.get("output").unwrap().clone();
        let db = map.get("grad_bias").unwrap().clone();
        let dx = map.get("grad_input").unwrap().clone();
        let dw = map.get("grad_weight").unwrap().clone();

        let b = b.reshape_no_alloc_owned([1, y.shape()[1], 1, 1]);
        let db = db.reshape_no_alloc_owned([1, y.shape()[1], 1, 1]);

        let config = ConvConfigs::new(
            x.shape_stride(),
            w.shape_stride(),
            &[1, 1],
            &[1, 1],
            &[1, 1],
        );

        let x = Variable::<f32, D>::new(x);
        let w = Variable::<f32, D>::new(w);
        let b = Variable::<f32, D>::new(b);

        let y_hat = conv(x.clone(), w.clone(), Some(b.clone()), config);
        assert_val_eq!(y_hat.clone(), y, 1e-4);
        y_hat.backward();
        assert_val_eq_grad!(x, dx, 1e-4);
        assert_val_eq_grad!(w, dw, 1e-4);
        assert_val_eq_grad!(b, db, 1e-4);
    }
    run_test!(conv_bias, conv_bias_cpu, conv_bias_gpu);
}

use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::DimDyn,
    nn::conv::{
        conv_bkwd_data, conv_bkwd_weight, conv_fwd,
        interface::{ConvBkwdDataConfig, ConvBkwdFilterConfig, ConvFwdConfig},
    },
    num::Num,
    shape_stride::ShapeStride,
};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

#[derive(Clone)]
pub struct ConvConfigs<T: Num> {
    fwd: Rc<RefCell<ConvFwdConfig<T>>>,
    bkwd_data: Rc<RefCell<ConvBkwdDataConfig<T>>>,
    bkwd_filter: Rc<RefCell<ConvBkwdFilterConfig<T>>>,
}

impl<T: Num> ConvConfigs<T> {
    #[must_use]
    pub fn new(
        input: ShapeStride<DimDyn>,
        filter: ShapeStride<DimDyn>,
        stride: &[usize],
        padding: &[usize],
        dilated: &[usize],
    ) -> Self {
        let fwd = ConvFwdConfig::new(
            input,
            filter,
            stride.to_vec(),
            padding.to_vec(),
            dilated.to_vec(),
        );
        let bkwd_data = ConvBkwdDataConfig::new(
            input,
            filter,
            stride.to_vec(),
            padding.to_vec(),
            dilated.to_vec(),
        );
        let bkwd_filter = ConvBkwdFilterConfig::new(
            input,
            filter,
            stride.to_vec(),
            padding.to_vec(),
            dilated.to_vec(),
        );
        let fwd = Rc::new(RefCell::new(fwd));
        let bkwd_data = Rc::new(RefCell::new(bkwd_data));
        let bkwd_filter = Rc::new(RefCell::new(bkwd_filter));
        Self {
            fwd,
            bkwd_data,
            bkwd_filter,
        }
    }

    #[must_use]
    pub fn get_output_shape(&self) -> ShapeStride<DimDyn> {
        self.fwd.borrow().output_shape()
    }

    #[must_use]
    pub fn get_input_shape(&self) -> ShapeStride<DimDyn> {
        self.fwd.borrow().input_shape()
    }

    #[must_use]
    pub fn get_filter_shape(&self) -> ShapeStride<DimDyn> {
        self.fwd.borrow().filter_shape()
    }
}

struct ConvFwd<T: Num, D: Device> {
    input: Variable<T, D>,
    filter: Variable<T, D>,
    output: VariableWeak<T, D>,
    config: ConvConfigs<T>,
}

struct Deconv<T: Num, D: Device> {
    d_output: Variable<T, D>,
    filter: Variable<T, D>,
    d_input: VariableWeak<T, D>,
    config: ConvConfigs<T>,
}

struct ConvBkwdFilter<T: Num, D: Device> {
    d_output: Variable<T, D>,
    input: Variable<T, D>,
    d_filter: VariableWeak<T, D>,
    config: ConvConfigs<T>,
}

impl<T: Num, D: Device> Function<T, D> for ConvFwd<T, D> {
    fn forward(&self) {
        conv_fwd(
            self.input.get_as_ref(),
            self.filter.get_as_ref(),
            self.output.upgrade().unwrap().get_as_mut(),
            &mut self.config.fwd.borrow_mut(),
        );
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().unwrap();
        let input_grad = deconv(
            output_grad.clone(),
            self.filter.clone(),
            self.config.clone(),
        );
        self.input.set_grad(input_grad);
        let filter_grad =
            conv_bkwd_filter(output_grad.clone(), self.input.clone(), self.config.clone());
        self.filter.set_grad(filter_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone(), self.filter.clone()]
    }
}

impl<T: Num, D: Device> Function<T, D> for Deconv<T, D> {
    fn forward(&self) {
        conv_bkwd_data(
            self.d_output.get_as_ref(),
            self.filter.get_as_ref(),
            self.d_input.upgrade().unwrap().get_as_mut(),
            &mut self.config.bkwd_data.borrow_mut(),
        );
    }

    fn backward(&self) {
        let input_grad = self.d_input.upgrade().unwrap().get_grad().unwrap();
        let output_grad = conv(input_grad.clone(), self.filter.clone(), self.config.clone());
        self.d_output.set_grad(output_grad);
        let filter_grad = conv_bkwd_filter(
            self.d_output.clone(),
            input_grad.clone(),
            self.config.clone(),
        );
        self.filter.set_grad(filter_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.d_output.clone(), self.filter.clone()]
    }
}

impl<T: Num, D: Device> Function<T, D> for ConvBkwdFilter<T, D> {
    fn forward(&self) {
        conv_bkwd_weight(
            self.d_output.get_as_ref(),
            self.input.get_as_ref(),
            self.d_filter.upgrade().unwrap().get_as_mut(),
            &mut self.config.bkwd_filter.borrow_mut(),
        );
    }

    fn backward(&self) {
        let filter_grad = self.d_filter.upgrade().unwrap().get_grad().unwrap();
        let output_grad = conv(self.input.clone(), filter_grad.clone(), self.config.clone());
        self.d_output.set_grad(output_grad);
        let input_grad = deconv(
            self.d_output.clone(),
            filter_grad.clone(),
            self.config.clone(),
        );
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.d_output.clone(), self.input.clone()]
    }
}

pub fn conv<T: Num, D: Device>(
    input: Variable<T, D>,
    filter: Variable<T, D>,
    config: ConvConfigs<T>,
) -> Variable<T, D> {
    let output_shape_stride = config.get_output_shape();
    let output = alloc(output_shape_stride.shape());
    let graph = ConvFwd {
        input,
        filter,
        output: output.clone().downgrade(),
        config,
    };
    graph.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(graph))));
    output
}

pub fn deconv<T: Num, D: Device>(
    d_output: Variable<T, D>,
    filter: Variable<T, D>,
    config: ConvConfigs<T>,
) -> Variable<T, D> {
    let input_shape_stride = config.get_input_shape();
    let input = alloc(input_shape_stride.shape());
    let graph = Deconv {
        d_output,
        filter,
        d_input: input.clone().downgrade(),
        config,
    };
    graph.forward();
    input.set_creator(Rc::new(RefCell::new(Box::new(graph))));
    input
}

pub fn conv_bkwd_filter<T: Num, D: Device>(
    d_output: Variable<T, D>,
    input: Variable<T, D>,
    config: ConvConfigs<T>,
) -> Variable<T, D> {
    let filter_shape_stride = config.get_filter_shape();
    let filter = alloc(filter_shape_stride.shape());
    let graph = ConvBkwdFilter {
        d_output,
        input,
        d_filter: filter.clone().downgrade(),
        config,
    };
    graph.forward();
    filter.set_creator(Rc::new(RefCell::new(Box::new(graph))));
    filter
}

#[cfg(test)]
mod conv_test {
    use std::collections::HashMap;

    use crate::Variable;
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, read_test_case_from_json_val, run_test};

    use super::{conv, ConvConfigs};

    fn conv2d<D: Device>() {
        let map = read_test_case_from_json_val!("../test_data_json/conv2d.json");
        let map: HashMap<String, Matrix<Owned<f32>, DimDyn, D>> = map
            .into_iter()
            .map(|(key, value)| (key, value.to()))
            .collect();
        let x = map.get("input").unwrap().clone();
        let w = map.get("filter").unwrap().clone();
        let y = map.get("output").unwrap().clone();
        let dx = map.get("grad_input").unwrap().clone();
        let dw = map.get("grad_weight").unwrap().clone();

        let config = ConvConfigs::new(
            x.shape_stride(),
            w.shape_stride(),
            &[1, 1],
            &[1, 1],
            &[1, 1],
        );

        let x = Variable::<f32, D>::new(x);
        let w = Variable::<f32, D>::new(w);
        let y_hat = conv(x.clone(), w.clone(), config);
        y_hat.backward();
        assert_val_eq!(y_hat, y, 1e-4);
        assert_val_eq_grad!(x, dx, 1e-4);
        assert_val_eq_grad!(w, dw, 1e-4);
    }
    run_test!(conv2d, conv2d_cpu, conv2d_gpu);
}

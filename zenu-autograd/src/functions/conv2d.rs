use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::DimTrait,
    nn::conv2d::{conv2d_bckwd_data, conv2d_bckwd_filter, conv2d_forward, conv2d_out_size},
    num::Num,
};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

struct Conv2d<T: Num, D: Device> {
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    bias: Option<Variable<T, D>>,
    y: VariableWeak<T, D>,
}

struct Deconv2d<T: Num, D: Device> {
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    y: VariableWeak<T, D>,
}

struct Conv2dBackward<T: Num, D: Device> {
    y_grad: Variable<T, D>,
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    filter_grad: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for Conv2d<T, D> {
    fn forward(&self) {
        let y = conv2d_forward(
            self.x.get_data().to_ref(),
            self.filter.get_data().to_ref(),
            // FIXME bias
            None,
            self.padding.0,
            self.padding.1,
            self.stride.0,
            self.stride.1,
            // FIXME use dilated
            1,
            1,
            None,
        );
        self.y
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&y);
    }

    fn backward(&self) {
        let gx = deconv2d(
            self.y.upgrade().unwrap().clone(),
            self.filter.clone(),
            self.stride,
            self.padding,
            None,
        );

        let gfilter = conv2d_filter_grad(
            self.x.clone(),
            self.y.upgrade().unwrap().clone(),
            self.stride,
            self.padding,
            self.filter.clone(),
        );

        self.x.set_grad(gx);
        self.filter.set_grad(gfilter);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.filter.clone()]
    }
}

impl<T: Num, D: Device> Function<T, D> for Deconv2d<T, D> {
    fn forward(&self) {
        let y = conv2d_bckwd_data(
            self.x.get_data().to_ref(),
            self.filter.get_data().to_ref(),
            self.padding.0,
            self.padding.1,
            self.stride.0,
            self.stride.1,
            1,
            1,
            None,
        );

        self.y
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&y);
    }

    fn backward(&self) {
        let gx = conv2d(
            self.y.upgrade().unwrap().clone(),
            self.filter.clone(),
            self.stride,
            self.padding,
            None,
        );

        let gfilter = conv2d_filter_grad(
            self.x.clone(),
            self.filter.clone(),
            self.stride,
            self.padding,
            self.filter.clone(),
        );

        self.x.set_grad(gx);
        self.filter.set_grad(gfilter);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.filter.clone()]
    }
}

impl<T: Num, D: Device> Function<T, D> for Conv2dBackward<T, D> {
    fn forward(&self) {
        let filter_grad = conv2d_bckwd_filter(
            self.x.get_data().to_ref(),
            self.filter.get_data().to_ref(),
            self.padding.0,
            self.padding.1,
            self.stride.0,
            self.stride.1,
            1,
            1,
            self.filter.get_data().shape(),
            None,
        );
        self.filter_grad
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&filter_grad);
    }

    fn backward(&self) {
        let gx = deconv2d(
            self.y_grad.clone(),
            self.filter.clone(),
            self.stride,
            self.padding,
            None,
        );

        let gfilter = conv2d(
            self.x.clone(),
            self.filter_grad.upgrade().unwrap(),
            self.stride,
            self.padding,
            None,
        );

        self.x.set_grad(gx);
        self.filter.set_grad(gfilter);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.filter.clone()]
    }
}

pub fn conv2d<T: Num, D: Device>(
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    bias: Option<Variable<T, D>>,
) -> Variable<T, D> {
    let conv2d_y_size = conv2d_out_size(
        x.get_data().shape().slice(),
        filter.get_data().shape().slice(),
        padding,
        stride,
    );
    let y = zeros(conv2d_y_size);
    let conv2d = Conv2d {
        x,
        filter,
        stride,
        padding,
        bias,
        y: y.clone().downgrade(),
    };
    conv2d.forward();
    y.set_creator(Rc::new(RefCell::new(Box::new(conv2d))));
    y
}

pub fn deconv2d<T: Num, D: Device>(
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    bias: Option<Variable<T, D>>,
) -> Variable<T, D> {
    let deconv2d_y_size = conv2d_out_size(
        x.get_data().shape().slice(),
        filter.get_data().shape().slice(),
        padding,
        stride,
    );
    let y = zeros(deconv2d_y_size);
    let deconv2d = Deconv2d {
        x,
        filter,
        stride,
        padding,
        y: y.clone().downgrade(),
    };
    deconv2d.forward();
    y.set_creator(Rc::new(RefCell::new(Box::new(deconv2d))));
    match bias {
        Some(bias) => y + bias,
        None => y,
    }
}

fn conv2d_filter_grad<T: Num, D: Device>(
    x: Variable<T, D>,
    y_grad: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    filter: Variable<T, D>,
) -> Variable<T, D> {
    let filter_grad = zeros(filter.get_data().shape().slice());
    let conv2d_bkwd_filter = Conv2dBackward {
        y_grad,
        x,
        filter,
        stride,
        padding,
        filter_grad: filter_grad.clone().downgrade(),
    };
    conv2d_bkwd_filter.forward();
    filter_grad.set_creator(Rc::new(RefCell::new(Box::new(conv2d_bkwd_filter))));
    filter_grad
}

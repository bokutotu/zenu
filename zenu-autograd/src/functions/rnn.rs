use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, nn::rnn::RNNDescriptor, num::Num};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

#[cfg(feature = "nvidia")]
use zenu_matrix::device::nvidia::Nvidia;

#[cfg(feature = "nvidia")]
struct CudnnRNN<T: Num> {
    rnn_desc: Rc<RefCell<RNNDescriptor<T>>>,
    x: Variable<T, Nvidia>,
    hx: Option<Variable<T, Nvidia>>,
    hy: VariableWeak<T, Nvidia>,
    y: VariableWeak<T, Nvidia>,
}

#[cfg(feature = "nvidia")]
impl<T: Num> Function<T, Nvidia> for CudnnRNN<T> {
    fn forward(&self) {
        let rnn_desc = self.rnn_desc.borrow_mut();
        rnn_desc.forward(
            &self.x,
            self.hx.as_ref(),
            &self.hy.upgrade().unwrap(),
            &self.y.upgrade().unwrap(),
        );
    }

    fn backward(&self) {
        let rnn_desc = self.rnn_desc.borrow();
        let mut rnn = rnn_desc.borrow_mut();
        rnn.backward(
            &self.x,
            self.hx.as_ref(),
            &self.hy.upgrade().unwrap(),
            &self.y.upgrade().unwrap(),
        );
    }

    fn get_inputs(&self) -> Vec<Variable<T, Nvidia>> {
        vec![self.x.clone(), self.hx.clone().unwrap()]
    }
}

pub struct RNNOutput<T: Num, D: Device> {
    pub y: Variable<T, D>,
    pub hy: Variable<T, D>,
}

pub fn cudnn_rnn_fwd<T: Num>(
    rnn_desc: Rc<RefCell<RNNDescriptor<T>>>,
    x: Variable<T, Nvidia>,
    hx: Option<Variable<T, Nvidia>>,
) -> RNNOutput<T, Nvidia> {
    let num_layers = rnn_desc.borrow().get_num_layers();
    let hidden_size = rnn_desc.borrow().get_hidden_size();
    let seq_len = x.shape()[0];
    let batch_size = x.shape()[1];
    let hy = alloc([num_layers, hidden_size]);
    let y = alloc([seq_len, batch_size, hidden_size]);
    let layer = CudnnRNN {
        rnn_desc,
        x,
        hx,
        hy: VariableWeak::downgrade(&hy),
        y: VariableWeak::downgrade(&y),
    };

    layer.forward();

    hy.set_creator(Rc::new(RefCell::new(Box::new(layer))));
    y.set_creator(Rc::new(RefCell::new(Box::new(layer))));

    RNNOutput { y, hy }
}

#[cfg(feature = "nvidia")]
struct CudnnRnnBkwd<T: Num> {
    rnn_desc: Rc<RefCell<RNNDescriptor<T>>>,
    x: Variable<T, Nvidia>,
    hx: Option<Variable<T, Nvidia>>,
    hy: Variable<T, Nvidia>,
    y: Variable<T, Nvidia>,
    dy: Variable<T, Nvidia>,
    dhy: Option<Variable<T, Nvidia>>,
    dx: VariableWeak<T, Nvidia>,
    dhx: VariableWeak<T, Nvidia>,
    dw: VariableWeak<T, Nvidia>,
}

#[cfg(feature = "nvidia")]
impl<T: Num> Function<T, Nvidia> for CudnnRnnBkwd<T> {
    fn forward(&mut self) {
        let rnn_desc = self.rnn_desc.borrow();
        let mut rnn = rnn_desc.borrow_mut();
        let dx = rnn.rnn_bkwd();
    }

    fn backward(&mut self) {
        unimplemented!(
            "this rnn fucntionn is use cudnn. cudnn rnn backward is not implemented yet"
        );
    }

    fn get_inputs(&self) -> Vec<Variable<T, Nvidia>> {
        vec![
            self.x.clone(),
            self.hx.clone().unwrap(),
            self.hy.clone(),
            self.y.clone(),
            self.dy.clone(),
            self.dhy.clone().unwrap(),
        ]
    }
}

use std::{cell::RefCell, ops::DerefMut, rc::Rc};

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
    is_training: bool,
}

#[cfg(feature = "nvidia")]
impl<T: Num> Function<T, Nvidia> for CudnnRNN<T> {
    fn forward(&self) {
        let mut rnn_desc = self.rnn_desc.borrow_mut();
        let rnn_desc = rnn_desc.deref_mut();
        let hx = val_option_to_ref_mat_option(&self.hx);
        let output = rnn_desc.rnn_fwd(self.x.get_data().to_ref(), hx, self.is_training);
        self.hy
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&output.hy);
        self.y
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&output.y);
    }

    fn backward(&self) {
        todo!();
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
    is_training: bool,
) -> RNNOutput<T, Nvidia> {
    let num_layers = rnn_desc.borrow().get_num_layers();
    let hidden_size = rnn_desc.borrow().get_hidden_size();
    let x_shape = x.get_shape();
    let seq_len = x_shape[0];
    let batch_size = x_shape[1];
    let hy = alloc([num_layers, hidden_size]);
    let y = alloc([seq_len, batch_size, hidden_size]);
    let layer = CudnnRNN {
        rnn_desc,
        x,
        hx,
        hy: hy.clone().downgrade(),
        y: y.clone().downgrade(),
        is_training,
    };

    layer.forward();

    let layer = Rc::new(RefCell::new(Box::new(layer) as Box<dyn Function<T, Nvidia>>));
    hy.set_creator(layer.clone());
    y.set_creator(layer);

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
    fn forward(&self) {
        let mut rnn_desc = self.rnn_desc.borrow_mut();
        let data = rnn_desc.rnn_bkwd_data(
            self.x.get_shape(),
            self.y.get_data().to_ref(),
            self.dy.get_data().to_ref(),
            self.hx.as_ref().map(|hx| hx.get_data().to_ref()),
            self.dhy.as_ref().map(|dhy| dhy.get_data().to_ref()),
        );
    }

    fn backward(&self) {
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

use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, nn::rnn::RNNDescriptor, num::Num};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

#[cfg(feature = "nvidia")]
use zenu_matrix::device::nvidia::Nvidia;

#[cfg(feature = "nvidia")]
struct CudnnRNN<T: Num> {
    rnn_desc: Rc<RefCell<RNNDescriptor<T>>>,
    x: Variable<T, Nvidia>,
    weight: Variable<T, Nvidia>,
    hx: Option<Variable<T, Nvidia>>,
    hy: VariableWeak<T, Nvidia>,
    y: VariableWeak<T, Nvidia>,
    is_training: bool,
}

#[cfg(feature = "nvidia")]
impl<T: Num> Function<T, Nvidia> for CudnnRNN<T> {
    fn forward(&self) {
        let mut rnn_desc = self.rnn_desc.borrow_mut();
        let hx = self.hx.as_ref().map(|hx| (*hx.get_data()).to_ref());
        let output = rnn_desc.rnn_fwd(
            self.x.get_as_ref(),
            hx,
            self.weight.get_as_ref(),
            self.is_training,
        );
        self.hy.upgrade().unwrap().swap_inner(output.hy);
        self.y.upgrade().unwrap().swap_inner(output.y);
    }

    fn backward(&self) {
        assert!(
            self.is_training,
            "backward is not allowed when is_training is false"
        );

        let mut rnn_desc = self.rnn_desc.borrow_mut();

        let x = self.x.get_as_ref();
        let y = self.y.upgrade().unwrap().get_as_ref();
        let dy = self.y.upgrade().unwrap().get_grad().unwrap().get_as_ref();
        let hx = self.hx.as_ref().map(|hx| (*hx.get_data()).to_ref());
        let dhy = self
            .hy
            .upgrade()
            .unwrap()
            .get_grad()
            .map(|dhy| dhy.get_as_ref());
        let weight = self.weight.get_data().to_ref();
        let ddata =
            rnn_desc.rnn_bkwd_data(self.x.get_shape(), y.clone(), dy, hx.clone(), dhy, weight);

        let dw = rnn_desc.rnn_bkwd_weights(x, hx, y);

        self.x.set_grad(Variable::new(ddata.dx));
        if let Some(hx) = self.hx.as_ref() {
            hx.set_grad(Variable::new(ddata.dhx));
        }

        self.weight.set_grad(Variable::new(dw));
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
    weight: Variable<T, Nvidia>,
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
        weight,
        is_training,
    };

    layer.forward();

    let layer = Rc::new(RefCell::new(Box::new(layer) as Box<dyn Function<T, Nvidia>>));
    hy.set_creator(layer.clone());
    y.set_creator(layer);

    RNNOutput { y, hy }
}

#[cfg(test)]
mod rnn {
    use zenu_matrix::nn::rnn::RNNDescriptor;

    #[test]
    fn rnn() {
        // let rnn_desc = RNNDescriptor::<f32>::new(RNNCell::RNN)
    }
}

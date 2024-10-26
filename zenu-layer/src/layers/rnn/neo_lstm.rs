use zenu_autograd::{
    nn::rnns::{
        lstm::{cudnn::lstm_cudnn, naive::lstm_naive},
        weights::LSTMCell,
    },
    Variable,
};
use zenu_matrix::{device::Device, num::Num};

use crate::ModuleParameters;

use super::neo_struct::NeoRNN;

pub struct LSTMInput<T: Num, D: Device> {
    pub x: Variable<T, D>,
    pub hx: Variable<T, D>,
    pub cx: Variable<T, D>,
}

impl<T: Num, D: Device> ModuleParameters<T, D> for LSTMInput<T, D> {}

impl<T: Num, D: Device> NeoRNN<T, D, LSTMCell> {
    fn forward(&self, input: LSTMInput<T, D>) -> Variable<T, D> {
        #[cfg(feature = "nvidia")]
        if self.is_cudnn {
            let desc = self.desc.as_ref().unwrap();
            let weights = self.cudnn_weights.as_ref().unwrap();

            let out = lstm_cudnn(
                desc.clone(),
                input.x.to(),
                Some(input.hx.to()),
                Some(input.cx.to()),
                weights.to(),
                self.is_training,
            );

            return out.to();
        }

        lstm_naive(
            input.x,
            input.hx,
            input.cx,
            self.weights.as_ref().unwrap(),
            self.is_bidirectional,
        )
    }
}

use zenu_autograd::{
    nn::rnns::{lstm::naive::lstm_naive, weights::LSTMCell},
    Variable,
};

#[cfg(feature = "nvidia")]
use zenu_autograd::nn::rnns::lstm::cudnn::lstm_cudnn;

use zenu_matrix::{device::Device, num::Num};

use crate::{Module, ModuleParameters, Parameters};

use super::{builder::RNNSLayerBuilder, inner::RNNInner};

pub struct LSTMInput<T: Num, D: Device> {
    pub x: Variable<T, D>,
    pub hx: Variable<T, D>,
    pub cx: Variable<T, D>,
}

impl<T: Num, D: Device> ModuleParameters<T, D> for LSTMInput<T, D> {}

impl<T: Num, D: Device> RNNInner<T, D, LSTMCell> {
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

pub struct LSTM<T: Num, D: Device>(RNNInner<T, D, LSTMCell>);

impl<T: Num, D: Device> Parameters<T, D> for LSTM<T, D> {
    fn weights(&self) -> std::collections::HashMap<String, Variable<T, D>> {
        self.0.weights()
    }

    fn biases(&self) -> std::collections::HashMap<String, Variable<T, D>> {
        self.0.biases()
    }

    fn load_parameters(&mut self, parameters: std::collections::HashMap<String, Variable<T, D>>) {
        self.0.load_parameters(parameters)
    }
}

impl<T: Num, D: Device> Module<T, D> for LSTM<T, D> {
    type Input = LSTMInput<T, D>;
    type Output = Variable<T, D>;

    fn call(&self, input: Self::Input) -> Self::Output {
        self.0.forward(input)
    }
}

pub type LSTMBuilder<T, D> = RNNSLayerBuilder<T, D, LSTMCell>;

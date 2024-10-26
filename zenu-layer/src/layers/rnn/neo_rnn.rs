use zenu_autograd::{
    nn::rnns::{
        rnn::{
            cudnn::cudnn_rnn_fwd,
            naive::{rnn_relu, rnn_tanh},
            RNNOutput,
        },
        weights::{CellType, RNNCell},
    },
    Variable,
};
use zenu_matrix::{device::Device, num::Num};

use crate::{Module, ModuleParameters, Parameters};
#[cfg(feature = "nvidia")]
use zenu_matrix::device::nvidia::Nvidia;

use super::neo_struct::{Activation, NeoRNN};

pub struct RNNLayerInput<T: Num, D: Device> {
    pub x: Variable<T, D>,
    pub hx: Variable<T, D>,
}

impl<T: Num, D: Device> ModuleParameters<T, D> for RNNLayerInput<T, D> {}

impl<T: Num, D: Device> NeoRNN<T, D, RNNCell> {
    fn forward(&self, input: RNNLayerInput<T, D>) -> Variable<T, D> {
        #[cfg(feature = "nvidia")]
        if self.is_cudnn {
            let desc = self.desc.as_ref().unwrap();
            let weights = self.cudnn_weights.as_ref().unwrap();

            let out: RNNOutput<T, Nvidia> = cudnn_rnn_fwd(
                desc.clone(),
                input.x.to(),
                Some(input.hx.to()),
                weights.to(),
                self.is_training,
            );

            return out.y.to();
        }

        let activation = self.activation.unwrap();
        if activation == Activation::ReLU {
            rnn_relu(
                input.x,
                input.hx,
                self.weights.as_ref().unwrap(),
                self.is_bidirectional,
            )
        } else {
            rnn_tanh(
                input.x,
                input.hx,
                self.weights.as_ref().unwrap(),
                self.is_bidirectional,
            )
        }
    }
}

pub struct RNN<T: Num, D: Device>(NeoRNN<T, D, RNNCell>);

impl<T: Num, D: Device> Parameters<T, D> for RNN<T, D> {
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

impl<T: Num, D: Device> Module<T, D> for RNN<T, D> {
    type Input = RNNLayerInput<T, D>;
    type Output = Variable<T, D>;

    fn call(&self, input: Self::Input) -> Self::Output {
        self.0.forward(input)
    }
}

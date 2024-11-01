use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::{
    nn::rnns::{gru::naive::gru_naive, weights::GRUCell},
    Variable,
};

#[cfg(feature = "nvidia")]
use zenu_autograd::nn::rnns::gru::cudnn::gru_cudnn;

use zenu_matrix::{device::Device, num::Num};

use crate::{Module, ModuleParameters, Parameters};

use super::{builder::RNNSLayerBuilder, inner::RNNInner};

pub struct GRUInput<T: Num, D: Device> {
    pub x: Variable<T, D>,
    pub hx: Variable<T, D>,
}

impl<T: Num, D: Device> ModuleParameters<T, D> for GRUInput<T, D> {}

impl<T: Num, D: Device> RNNInner<T, D, GRUCell> {
    fn forward(&self, input: GRUInput<T, D>) -> Variable<T, D> {
        #[cfg(feature = "nvidia")]
        if self.is_cudnn {
            let desc = self.desc.as_ref().unwrap();
            let weights = self.cudnn_weights.as_ref().unwrap();

            let out = gru_cudnn(
                desc.clone(),
                input.x.to(),
                Some(input.hx.to()),
                weights.to(),
                self.is_training,
            );

            return out.y.to();
        }

        gru_naive(
            input.x,
            input.hx,
            self.weights.as_ref().unwrap(),
            self.is_bidirectional,
        )
    }
}

pub struct GRU<T: Num, D: Device>(RNNInner<T, D, GRUCell>);

impl<T: Num, D: Device> Parameters<T, D> for GRU<T, D> {
    fn weights(&self) -> std::collections::HashMap<String, Variable<T, D>> {
        self.0.weights()
    }

    fn biases(&self) -> std::collections::HashMap<String, Variable<T, D>> {
        self.0.biases()
    }

    fn load_parameters(&mut self, parameters: std::collections::HashMap<String, Variable<T, D>>) {
        self.0.load_parameters(parameters);
    }
}

impl<T: Num, D: Device> Module<T, D> for GRU<T, D> {
    type Input = GRUInput<T, D>;
    type Output = Variable<T, D>;

    fn call(&self, input: Self::Input) -> Self::Output {
        self.0.forward(input)
    }
}

pub type GRUBuilder<T, D> = RNNSLayerBuilder<T, D, GRUCell>;

impl<T: Num, D: Device> RNNSLayerBuilder<T, D, GRUCell>
where
    StandardNormal: Distribution<T>,
{
    pub fn build_gru(self) -> GRU<T, D> {
        GRU(self.build_inner())
    }
}

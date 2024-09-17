use zenu_autograd::functions::rnn::naive::RNNLayerWeights;
use zenu_matrix::{device::Device, num::Num};

#[cfg(feature = "nvidia")]
use zenu_autograd::rnn::cudnn::CudnnRNN;

pub struct RNN<T: Num, D: Device> {
    weights: Vec<RNNLayerWeights<T, D>>,
    #[cfg(feature = "nvidia")]
    config: CudnnRNN<T>,
    is_cudnn: bool,
    is_bidirectional: bool,
}

#[derive(Debug, Default)]
pub struct RNNBuilder<T: Num, D: Device> {
    is_cudnn: bool,
    is_bidirectional: bool,
    hidden_size: usize,
    num_layers: usize,
    input_size: usize,
}

impl<T: Num, D: Device> RNNBuilder<T, D> {
    pub fn is_cudnn(mut self, is_cudnn: bool) -> Self {
        self.is_cudnn = is_cudnn;
        self
    }

    pub fn is_bidirectional(mut self, is_bidirectional: bool) -> Self {
        self.is_bidirectional = is_bidirectional;
        self
    }

    pub fn hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = input_size;
        self
    }

    pub fn build(self) -> RNN<T, D> {
        let mut weights = Vec::with_capacity(self.num_layers);
        for _ in 0..self.num_layers {
            weights.push(RNNLayerWeights::new(self.hidden_size, self.input_size));
        }
        let config = CudnnRNN::new(
            self.num_layers,
            self.hidden_size,
            self.input_size,
            self.is_bidirectional,
        );
        RNN {
            weights,
            config,
            is_cudnn: self.is_cudnn,
            is_bidirectional: self.is_bidirectional,
        }
    }
}

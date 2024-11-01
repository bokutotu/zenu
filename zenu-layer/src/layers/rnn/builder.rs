#[cfg(feature = "nvidia")]
use std::{cell::RefCell, rc::Rc};

use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::nn::rnns::weights::CellType;
use zenu_matrix::{device::Device, num::Num};

#[cfg(feature = "nvidia")]
use zenu_matrix::{device::nvidia::Nvidia, nn::rnn::RNNDescriptor};

use zenu_autograd::nn::rnns::weights::{RNNCell, RNNLayerWeights};

#[cfg(feature = "nvidia")]
use zenu_autograd::{creator::alloc::alloc, Variable};

#[cfg(feature = "nvidia")]
use crate::layers::rnn::inner::rnn_weights_to_desc;

use crate::layers::rnn::inner::{Activation, RNNInner};

#[expect(clippy::module_name_repetitions)]
#[derive(Debug, Default)]
pub struct RNNSLayerBuilder<T: Num, D: Device, C: CellType> {
    is_cudnn: Option<bool>,
    is_bidirectional: Option<bool>,
    hidden_size: Option<usize>,
    num_layers: Option<usize>,
    input_size: Option<usize>,
    activation: Option<Activation>,
    batch_size: Option<usize>,
    is_training: Option<bool>,
    _type: std::marker::PhantomData<(T, D, C)>,
}

impl<T: Num, D: Device, C: CellType> RNNSLayerBuilder<T, D, C> {
    #[must_use]
    pub fn set_is_cudnn(mut self, is_cudnn: bool) -> Self {
        self.is_cudnn = Some(is_cudnn);
        self
    }

    #[must_use]
    pub fn set_is_bidirectional(mut self, is_bidirectional: bool) -> Self {
        self.is_bidirectional = Some(is_bidirectional);
        self
    }

    #[must_use]
    pub fn hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = Some(hidden_size);
        self
    }

    #[must_use]
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = Some(num_layers);
        self
    }

    #[must_use]
    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = Some(input_size);
        self
    }

    #[must_use]
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    #[must_use]
    pub fn set_is_training(mut self, is_training: bool) -> Self {
        self.is_training = Some(is_training);
        self
    }

    fn init_weight(&self, idx: usize) -> RNNLayerWeights<T, D, C>
    where
        StandardNormal: Distribution<T>,
    {
        let input_size = if self.is_bidirectional.unwrap() {
            if idx == 0 {
                self.input_size.unwrap()
            } else {
                self.hidden_size.unwrap() * 2
            }
        } else if idx == 0 {
            self.input_size.unwrap()
        } else {
            self.hidden_size.unwrap()
        };

        RNNLayerWeights::init(
            input_size,
            self.hidden_size.unwrap(),
            self.is_bidirectional.unwrap(),
        )
    }

    fn init_weights(&self) -> Vec<RNNLayerWeights<T, D, C>>
    where
        StandardNormal: Distribution<T>,
    {
        (0..self.num_layers.unwrap())
            .map(|idx| self.init_weight(idx))
            .collect()
    }

    #[cfg(feature = "nvidia")]
    fn init_cudnn_desc(&self) -> RNNDescriptor<T> {
        if self.activation.unwrap() == Activation::ReLU {
            RNNDescriptor::<T>::new_rnn_relu(
                self.is_bidirectional.unwrap(),
                0.0,
                self.input_size.unwrap(),
                self.hidden_size.unwrap(),
                self.num_layers.unwrap(),
                self.batch_size.unwrap(),
            )
        } else {
            RNNDescriptor::<T>::new_rnn_tanh(
                self.is_bidirectional.unwrap(),
                0.0,
                self.input_size.unwrap(),
                self.hidden_size.unwrap(),
                self.num_layers.unwrap(),
                self.batch_size.unwrap(),
            )
        }
    }

    #[cfg(feature = "nvidia")]
    fn load_cudnn_weights(
        &self,
        desc: &RNNDescriptor<T>,
        weights: Vec<RNNLayerWeights<T, D, C>>,
    ) -> Variable<T, Nvidia> {
        let cudnn_weight_bytes = desc.get_weight_num_elems();
        let cudnn_weight: Variable<T, Nvidia> = alloc([cudnn_weight_bytes]);

        let weights =
            rnn_weights_to_desc::<T, D, C>(weights, self.is_bidirectional.unwrap_or(false));

        desc.load_rnn_weights(cudnn_weight.get_as_mut().as_mut_ptr().cast(), weights)
            .unwrap();

        cudnn_weight
    }

    #[must_use]
    pub(super) fn build_inner(mut self) -> RNNInner<T, D, C>
    where
        StandardNormal: Distribution<T>,
    {
        self.is_cudnn.get_or_insert(false);
        self.is_bidirectional.get_or_insert(false);
        self.activation.get_or_insert(Activation::ReLU);
        self.is_training.get_or_insert(true);
        assert!(self.hidden_size.is_some(), "hidden_size is required");
        assert!(self.num_layers.is_some(), "num_layers is required");
        assert!(self.input_size.is_some(), "input_size is required");
        assert!(self.batch_size.is_some(), "batch_size is required");

        #[cfg(not(feature = "nvidia"))]
        assert!(
            !self.is_cudnn.unwrap(),
            "cudnn is not enabled, please enable cudnn feature"
        );

        let weights = self.init_weights();
        #[cfg(feature = "nvidia")]
        if self.is_cudnn.unwrap() {
            let desc = self.init_cudnn_desc();
            let cudnn_weights = self.load_cudnn_weights(&desc, weights);
            let cudnn_weights = cudnn_weights.to::<Nvidia>();
            return RNNInner {
                weights: None,
                desc: Some(Rc::new(RefCell::new(desc))),
                cudnn_weights: Some(cudnn_weights),
                is_cudnn: self.is_cudnn.unwrap(),
                is_bidirectional: self.is_bidirectional.unwrap(),
                activation: self.activation,
                is_training: self.is_training.unwrap(),
            };
        }
        RNNInner {
            weights: Some(weights),
            #[cfg(feature = "nvidia")]
            desc: None,
            #[cfg(feature = "nvidia")]
            cudnn_weights: None,
            #[cfg(feature = "nvidia")]
            is_cudnn: self.is_cudnn.unwrap(),
            is_bidirectional: self.is_bidirectional.unwrap(),
            activation: self.activation,
            #[cfg(feature = "nvidia")]
            is_training: self.is_training.unwrap(),
        }
    }
}

impl<T: Num, D: Device> RNNSLayerBuilder<T, D, RNNCell> {
    #[must_use]
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = Some(activation);
        self
    }
}

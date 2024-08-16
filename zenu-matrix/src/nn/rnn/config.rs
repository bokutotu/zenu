use std::{cell::RefCell, rc::Rc};

use zenu_cuda::cudnn::rnn::{RNNAlgo, RNNBias, RNNCell, RNNMathType};
#[cfg(feature = "nvidia")]
use zenu_cuda::cudnn::rnn::{RNNConfig as NvidiaRNNConfig, RNNExecutor};

use crate::num::Num;

#[cfg(feature = "nvidia")]
pub struct NvidiaConfig<'a, T: Num> {
    pub config: NvidiaRNNConfig<T>,
    pub exe: Option<RNNExecutor<'a, T>>,
}

impl<T: Num> NvidiaConfig<'_, T> {
    fn new(
        cell: RNNCell,
        bidirectional: bool,
        dropout: f64,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_size: usize,
    ) -> Self {
        if dropout != 0.0 {
            panic!("Dropout is not supported in this version");
        }
        let config = NvidiaRNNConfig::new(
            RNNAlgo::Standard,
            cell,
            RNNBias::DoubleBias,
            bidirectional,
            RNNMathType::TensorOpAllowConversion,
            None,
            input_size,
            hidden_size,
            num_layers,
            batch_size,
        );
        Self { config, exe: None }
    }

    pub fn new_rnn_relu(
        bidirectional: bool,
        dropout: f64,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_size: usize,
    ) -> Self {
        Self::new(
            RNNCell::RNNRelu,
            bidirectional,
            dropout,
            input_size,
            hidden_size,
            num_layers,
            batch_size,
        )
    }

    pub fn new_rnn_tanh(
        bidirectional: bool,
        dropout: f64,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_size: usize,
    ) -> Self {
        Self::new(
            RNNCell::RNNTanh,
            bidirectional,
            dropout,
            input_size,
            hidden_size,
            num_layers,
            batch_size,
        )
    }

    pub fn lstm(
        bidirectional: bool,
        dropout: f64,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_size: usize,
    ) -> Self {
        Self::new(
            RNNCell::LSTM,
            bidirectional,
            dropout,
            input_size,
            hidden_size,
            num_layers,
            batch_size,
        )
    }

    pub fn gru(
        bidirectional: bool,
        dropout: f64,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_size: usize,
    ) -> Self {
        Self::new(
            RNNCell::GRU,
            bidirectional,
            dropout,
            input_size,
            hidden_size,
            num_layers,
            batch_size,
        )
    }
}

#[derive(Clone)]
pub struct RNNConfig<'a, T: Num> {
    #[cfg(feature = "nvidia")]
    pub config: Rc<RefCell<NvidiaConfig<'a, T>>>,

    _phantom: std::marker::PhantomData<T>,
}

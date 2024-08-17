use zenu_cuda::cudnn::rnn::{
    RNNAlgo, RNNBias, RNNCell, RNNConfig as NvidiaRNNConfig, RNNDataLayout, RNNExecutor,
    RNNMathType,
};

use crate::num::Num;

pub struct RNNConfig<T: Num> {
    pub config: NvidiaRNNConfig<T>,
}

impl<T: Num> RNNConfig<T> {
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
        Self { config }
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

    pub fn get_weight_bytes(&self) -> usize {
        self.config.weights_size
    }

    pub fn create_executor(&self, is_training: bool, seq_length: usize) -> RNNExecutor<T> {
        let seq_length_array = vec![seq_length; self.config.batch_size];
        RNNExecutor::new(
            &self.config,
            seq_length,
            &seq_length_array,
            RNNDataLayout::SeqMajorUnpacked,
            T::zero(),
            is_training,
        )
    }
}

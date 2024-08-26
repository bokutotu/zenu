use zenu_cuda::cudnn::rnn::{
    RNNAlgo, RNNBias, RNNCell, RNNContext, RNNDataLayout, RNNDescriptor as NvidiaRNNConfig,
    RNNMathType,
};

use crate::{device::Device, num::Num};

use super::RNNWeights;

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
            RNNMathType::TensorOp,
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

    pub fn create_executor(&self, is_training: bool, seq_length: usize) -> RNNContext<T> {
        let seq_length_array = vec![seq_length; self.config.batch_size];
        RNNContext::new(
            &self.config,
            seq_length,
            &seq_length_array,
            RNNDataLayout::SeqMajorUnpacked,
            T::zero(),
            is_training,
        )
    }

    pub fn get_input_size(&self) -> usize {
        self.config.input_size
    }

    pub fn get_hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub fn get_num_layers(&self) -> usize {
        self.config.num_layers
    }

    pub fn get_batch_size(&self) -> usize {
        self.config.batch_size
    }

    pub fn get_is_bidirectional(&self) -> bool {
        self.config.bidirectional
    }

    /// this function not check shape of params
    /// make sure that params has the same shape as the config
    pub fn load_rnn_weights<D: Device>(
        &self,
        ptr: *mut u8,
        params: Vec<RNNWeights<T, D>>,
    ) -> Result<(), String> {
        if self.get_num_layers() != params.len() {
            return Err("Number of layers does not match".to_string());
        }

        let rnn_params = self.config.get_rnn_params(ptr as *mut _);

        for idx in 0..self.get_num_layers() {
            let layer = &params[idx];
            let layer_params = &rnn_params[idx];

            layer.set_weight(layer_params);
        }

        Ok(())
    }
}

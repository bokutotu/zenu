use zenu_cuda::cudnn::rnn::{
    RNNAlgo, RNNBias, RNNCell, RNNDataLayout, RNNDescriptor as RNNDesc, RNNMathType,
};

use crate::{
    device::{nvidia::Nvidia, Device, DeviceBase},
    matrix::Matrix,
    num::Num,
};

use super::RNNWeights;

pub struct RNNDescriptor<T: Num> {
    pub desc: RNNDesc<T>,
    workspace: *mut u8,
    reserve_space: *mut u8,
}

impl<T: Num> Drop for RNNDescriptor<T> {
    fn drop(&mut self) {
        if !self.workspace.is_null() {
            Nvidia::drop_ptr(self.workspace);
        }
        if !self.reserve_space.is_null() {
            Nvidia::drop_ptr(self.reserve_space);
        }
    }
}

impl<T: Num> RNNDescriptor<T> {
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
        let desc = RNNDesc::new(
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
        Self {
            desc,
            workspace: std::ptr::null_mut(),
            reserve_space: std::ptr::null_mut(),
        }
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
        self.desc.get_weights_size()
    }

    /// this function set input sequence length
    /// if seq_length is different from the previous one, it will reallocate workspace
    pub fn config_seq_length(&mut self, is_training: bool, seq_length: usize) {
        let prev_workspace_size = if self.workspace.is_null() {
            0
        } else {
            self.desc.get_workspace_size()
        };
        let prev_reserve_space_size = if self.reserve_space.is_null() {
            0
        } else {
            self.desc.get_reserve_size()
        };

        self.desc.set_input_size(
            seq_length,
            &vec![seq_length; self.desc.get_num_layers()],
            RNNDataLayout::SeqMajorUnpacked,
            is_training,
            T::zero(),
        );

        if prev_workspace_size != self.desc.get_workspace_size()
            || prev_reserve_space_size != self.desc.get_reserve_size()
        {
            self.allocate_workspace();
        }
    }

    pub fn allocate_workspace(&mut self) {
        if !self.workspace.is_null() {
            Nvidia::drop_ptr(self.workspace);
        }
        if !self.reserve_space.is_null() {
            Nvidia::drop_ptr(self.reserve_space);
        }

        self.workspace = Nvidia::alloc(self.desc.get_workspace_size()).unwrap();
        self.reserve_space = Nvidia::alloc(self.desc.get_reserve_size()).unwrap();
    }

    pub fn get_input_size(&self) -> usize {
        self.desc.get_input_size()
    }

    pub fn get_hidden_size(&self) -> usize {
        self.desc.get_hidden_size()
    }

    pub fn get_num_layers(&self) -> usize {
        self.desc.get_num_layers()
    }

    pub fn get_batch_size(&self) -> usize {
        self.desc.get_batch_size()
    }

    pub fn get_is_bidirectional(&self) -> bool {
        self.desc.get_is_bidirectional()
    }

    /// this function not check shape of params
    /// make sure that params has the same shape as the config
    // TODO: ptrはRNNDescriptorのメンバにする
    pub fn load_rnn_weights<D: Device>(
        &self,
        ptr: *mut u8,
        params: Vec<RNNWeights<T, D>>,
    ) -> Result<(), String> {
        if self.get_num_layers() != params.len() {
            return Err("Number of layers does not match".to_string());
        }

        let rnn_params = self.desc.get_rnn_params(ptr as *mut _);

        for idx in 0..self.get_num_layers() {
            let layer = &params[idx];
            let layer_params = &rnn_params[idx];

            layer.set_weight(layer_params);
        }

        Ok(())
    }

    pub fn store_rnn_weights<D: Device>(&self, weight_ptr: *mut u8) -> Vec<RNNWeights<T, D>> {
        let mut params = Vec::with_capacity(self.get_num_layers());

        let rnn_params = self.desc.get_rnn_params(weight_ptr as *mut _);

        for idx in 0..self.get_num_layers() {
            let input_weight = if idx == 0 || (idx == 1 && self.get_is_bidirectional()) {
                let input_shape = &[self.get_hidden_size(), self.get_input_size()];
                Matrix::alloc(input_shape)
            } else {
                let input_shape = &[self.get_hidden_size(), self.get_hidden_size()];
                Matrix::alloc(input_shape)
            };
            let hidden_weight = Matrix::alloc(&[self.get_hidden_size(), self.get_hidden_size()]);
            let input_bias = Matrix::alloc(&[self.get_hidden_size()]);
            let hidden_bias = Matrix::alloc(&[self.get_hidden_size()]);

            let layer_params = RNNWeights::new(
                input_weight,
                hidden_weight,
                Some(input_bias),
                Some(hidden_bias),
            );

            let layer = &rnn_params[idx];
            layer_params.set_weight(layer);
            params.push(layer_params);
        }

        params
    }

    pub fn fwd(
        &self,
        x: *const T,
        y: *mut T,
        hx: *const T,
        hy: *mut T,
        cx: *const T,
        cy: *mut T,
        weight: *mut T,
    ) {
        self.desc.fwd(
            x,
            y,
            hx,
            hy,
            cx,
            cy,
            weight,
            self.workspace as *mut _,
            self.reserve_space as *mut _,
        );
    }

    pub fn bkwd_data(
        &self,
        y: *const T,
        dy: *const T,
        dx: *mut T,
        hx: *const T,
        dhy: *const T,
        dhx: *mut T,
        cx: *const T,
        dcy: *const T,
        dcx: *mut T,
        weight: *const T,
    ) {
        self.desc.bkwd_data(
            y,
            dy,
            dx,
            hx,
            dhy,
            dhx,
            cx,
            dcy,
            dcx,
            weight,
            self.workspace as *mut _,
            self.reserve_space as *mut _,
        )
    }

    pub fn bkwd_weights(&self, x: *const T, hx: *const T, y: *const T, dweight: *mut T) {
        self.desc.bkwd_weights(
            x,
            hx,
            y,
            dweight,
            self.workspace as *mut _,
            self.reserve_space as *mut _,
        )
    }
}

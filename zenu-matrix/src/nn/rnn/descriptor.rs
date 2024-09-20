use zenu_cuda::cudnn::rnn::{
    RNNAlgo, RNNBias, RNNCell, RNNDataLayout, RNNDescriptor as RNNDesc, RNNMathType,
};

use crate::{
    device::{nvidia::Nvidia, Device, DeviceBase},
    matrix::Matrix,
    nn::rnn::params::Params,
    num::Num,
};

use super::RNNWeightsMat;

pub struct Descriptor<T: Num, P: Params> {
    pub desc: RNNDesc<T>,
    workspace: Option<*mut u8>,
    reserve_space: Option<*mut u8>,
    _phantom: std::marker::PhantomData<P>,
}

pub type RNNDescriptor<T> = Descriptor<T, RNNWeightsMat<T, Nvidia>>;

impl<T: Num, P: Params> Drop for Descriptor<T, P> {
    fn drop(&mut self) {
        if self.workspace.is_some() {
            Nvidia::drop_ptr(self.workspace.unwrap());
        }
        if self.reserve_space.is_some() {
            Nvidia::drop_ptr(self.reserve_space.unwrap());
        }
    }
}

impl<T: Num, P: Params> Descriptor<T, P> {
    fn new(
        cell: RNNCell,
        bidirectional: bool,
        dropout: f64,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_size: usize,
    ) -> Self {
        assert!(dropout == 0.0, "Dropout is not supported in this version");
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
            workspace: None,
            reserve_space: None,
            _phantom: std::marker::PhantomData,
        }
    }

    #[must_use]
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

    #[must_use]
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

    #[must_use]
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

    #[must_use]
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

    pub(super) fn get_weight_bytes(&self) -> usize {
        self.desc.get_weights_size()
    }

    #[must_use]
    pub fn get_weight_num_elems(&self) -> usize {
        self.get_weight_bytes() / std::mem::size_of::<T>()
    }

    /// this function set input sequence length
    /// if `seq_length` is different from the previous one, it will reallocate workspace
    pub fn config_seq_length(&mut self, is_training: bool, seq_length: usize, batch_size: usize) {
        let prev_workspace_size = if self.workspace.is_none() {
            0
        } else {
            self.desc.get_workspace_size()
        };
        let prev_reserve_space_size = if self.reserve_space.is_none() {
            0
        } else {
            self.desc.get_reserve_size()
        };

        self.desc.set_input_size(
            seq_length,
            &vec![seq_length; batch_size],
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

    #[expect(clippy::missing_panics_doc)]
    pub fn allocate_workspace(&mut self) {
        if self.workspace.is_some() {
            Nvidia::drop_ptr(self.workspace.unwrap());
        }
        if self.reserve_space.is_some() {
            Nvidia::drop_ptr(self.reserve_space.unwrap());
        }

        self.workspace = Some(Nvidia::alloc(self.desc.get_workspace_size()).unwrap());
        self.reserve_space = Some(Nvidia::alloc(self.desc.get_reserve_size()).unwrap());
    }

    #[must_use]
    pub fn get_input_size(&self) -> usize {
        self.desc.get_input_size()
    }

    #[must_use]
    pub fn get_hidden_size(&self) -> usize {
        self.desc.get_hidden_size()
    }

    #[must_use]
    pub fn get_num_layers(&self) -> usize {
        self.desc.get_num_layers()
    }

    #[must_use]
    pub fn get_batch_size(&self) -> usize {
        self.desc.get_batch_size()
    }

    #[must_use]
    pub fn get_is_bidirectional(&self) -> bool {
        self.desc.get_is_bidirectional()
    }

    /// this function not check shape of params
    /// make sure that params has the same shape as the config
    #[expect(clippy::missing_errors_doc)]
    pub fn load_rnn_weights<D: Device>(
        &self,
        weight_ptr: *mut u8,
        mut params: Vec<RNNWeightsMat<T, D>>,
    ) -> Result<(), String> {
        let expected_size = self.get_num_layers() * if self.get_is_bidirectional() { 2 } else { 1 };

        if expected_size != params.len() {
            return Err("Number of layers does not match".to_string());
        }

        let rnn_params = self.desc.get_rnn_params(weight_ptr.cast());

        for idx in 0..params.len() {
            let layer = &mut params[idx];
            let layer_params = &rnn_params[idx];

            layer.set_weight(layer_params);
        }

        Ok(())
    }

    pub fn store_rnn_weights<D: Device>(&self, weight_ptr: *mut u8) -> Vec<RNNWeightsMat<T, D>> {
        let num_layers = self.get_num_layers() * if self.get_is_bidirectional() { 2 } else { 1 };
        let mut params = Vec::with_capacity(num_layers);

        let rnn_params = self.desc.get_rnn_params(weight_ptr.cast());

        for (idx, layer) in rnn_params.iter().enumerate() {
            let input_weight = if idx == 0 || (idx == 1 && self.get_is_bidirectional()) {
                let input_shape = &[self.get_hidden_size(), self.get_input_size()];
                Matrix::alloc(input_shape)
            } else {
                let input_len =
                    self.get_hidden_size() * if self.get_is_bidirectional() { 2 } else { 1 };
                let input_shape = &[self.get_hidden_size(), input_len];
                Matrix::alloc(input_shape)
            };
            let hidden_weight = Matrix::alloc([self.get_hidden_size(), self.get_hidden_size()]);
            let input_bias = Matrix::alloc([self.get_hidden_size()]);
            let hidden_bias = Matrix::alloc([self.get_hidden_size()]);

            let mut layer_params =
                RNNWeightsMat::new(input_weight, hidden_weight, input_bias, hidden_bias);

            layer_params.load_from_params(layer);
            params.push(layer_params);
        }

        params
    }

    #[expect(clippy::too_many_arguments)]
    pub(crate) fn fwd(
        &self,
        x: *const T,
        y: *mut T,
        hx: *const T,
        hy: *mut T,
        cx: *const T,
        cy: *mut T,
        weight: *const T,
    ) {
        self.desc.fwd(
            x,
            y,
            hx,
            hy,
            cx,
            cy,
            weight.cast_mut(),
            self.workspace.unwrap().cast(),
            self.reserve_space.unwrap().cast(),
        );
    }

    #[expect(clippy::too_many_arguments, clippy::similar_names)]
    pub(crate) fn bkwd_data(
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
            weight.cast_mut(),
            self.workspace.unwrap().cast(),
            self.reserve_space.unwrap().cast(),
        );
    }

    pub(crate) fn bkwd_weights(&self, x: *const T, hx: *const T, y: *const T, dweight: *mut T) {
        self.desc.bkwd_weights(
            x,
            hx,
            y,
            dweight,
            self.workspace.unwrap().cast(),
            self.reserve_space.unwrap().cast(),
        );
    }
}

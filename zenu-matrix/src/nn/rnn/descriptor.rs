use zenu_cuda::cudnn::rnn::{
    RNNAlgo, RNNBias, RNNCell, RNNDataLayout, RNNDescriptor as RNNDesc, RNNMathType,
};

use crate::{
    device::{nvidia::Nvidia, Device, DeviceBase},
    dim::DimDyn,
    matrix::Matrix,
    num::Num,
};

use super::RNNWeightsMat;

pub struct RNNDescriptor<T: Num> {
    pub desc: RNNDesc<T>,
    workspace: Option<*mut u8>,
    reserve_space: Option<*mut u8>,
}

impl<T: Num> Drop for RNNDescriptor<T> {
    fn drop(&mut self) {
        if self.workspace.is_some() {
            Nvidia::drop_ptr(self.workspace.unwrap());
        }
        if self.reserve_space.is_some() {
            Nvidia::drop_ptr(self.reserve_space.unwrap());
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

    /// the hidden size of the output is `hidden_size` * `num_directions`
    #[must_use]
    pub fn get_output_size(&self) -> usize {
        self.desc.get_hidden_size()
            * if self.desc.get_is_bidirectional() {
                2
            } else {
                1
            }
    }

    /// the number of layers of the output is `num_layers` * `num_directions`
    #[must_use]
    pub fn get_output_num_layers(&self) -> usize {
        self.desc.get_num_layers()
            * if self.desc.get_is_bidirectional() {
                2
            } else {
                1
            }
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

    fn weight_size_factor(&self) -> usize {
        match self.desc.get_cell() {
            RNNCell::RNNRelu | RNNCell::RNNTanh => 1,
            RNNCell::LSTM => 4,
            RNNCell::GRU => 3,
        }
    }

    fn cal_rnn_input_weight_size(&self, idx: usize) -> DimDyn {
        let factor = self.weight_size_factor();

        let input_size = if idx == 0 || (idx == 1 && self.get_is_bidirectional()) {
            self.get_input_size()
        } else {
            self.get_output_size()
        };

        [self.get_hidden_size() * factor, input_size].into()
    }

    fn call_rnn_hidden_weight_size(&self) -> DimDyn {
        let factor = self.weight_size_factor();
        [self.get_hidden_size() * factor, self.get_hidden_size()].into()
    }

    fn call_bias_size(&self) -> DimDyn {
        let factor = self.weight_size_factor();
        [self.get_hidden_size() * factor].into()
    }

    pub fn store_rnn_weights<D: Device>(&self, weight_ptr: *mut u8) -> Vec<RNNWeightsMat<T, D>> {
        let num_layers = self.get_output_num_layers();
        let mut params = Vec::with_capacity(num_layers);

        let rnn_params = self.desc.get_rnn_params(weight_ptr.cast());

        for (idx, layer) in rnn_params.iter().enumerate() {
            let input_weight_shape = self.cal_rnn_input_weight_size(idx);
            let hidden_weight_shape = self.call_rnn_hidden_weight_size();
            let bias_shape = self.call_bias_size();
            let input_weight = Matrix::alloc(input_weight_shape);
            let hidden_weight = Matrix::alloc(hidden_weight_shape);
            let input_bias = Matrix::alloc(bias_shape);
            let hidden_bias = Matrix::alloc(bias_shape);

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

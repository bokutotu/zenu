use zenu_cudnn_sys::{
    cudnnDropoutDescriptor_t, cudnnForwardMode_t, cudnnGetRNNTempSpaceSizes,
    cudnnRNNDataDescriptor_t, cudnnRNNDescriptor_t, cudnnTensorDescriptor_t,
};

use crate::{cudnn::tensor_descriptor_nd, ZENU_CUDA_STATE};

use super::{
    function::{
        rnn_bkwd_data, rnn_bkwd_weight, rnn_data_descriptor, rnn_descriptor, rnn_fwd,
        rnn_weight_space,
    },
    RNNAlgo, RNNBias, RNNCell, RNNDataLayout, RNNMathType,
};

pub struct RnnConfig<T: 'static> {
    pub rnn_desc: cudnnRNNDescriptor_t,
    pub h_desc: cudnnTensorDescriptor_t,
    pub c_desc: cudnnTensorDescriptor_t,
    pub weights_size: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    _marker: std::marker::PhantomData<T>,
}

pub struct RnnWorkspace {
    pub workspace_size: usize,
    pub reserve_size: usize,
}

impl<T: 'static> RnnConfig<T> {
    pub fn new(
        algo: RNNAlgo,
        cell: RNNCell,
        bias: RNNBias,
        bidirectional: bool,
        math: RNNMathType,
        dropout: Option<cudnnDropoutDescriptor_t>,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        batch_size: usize,
    ) -> Self {
        let h_desc = tensor_descriptor_nd::<T>(
            &[num_layers as i32, batch_size as i32, hidden_size as i32],
            &[(batch_size * hidden_size) as i32, hidden_size as i32, 1],
        )
        .unwrap();

        let c_desc = tensor_descriptor_nd::<T>(
            &[num_layers as i32, batch_size as i32, hidden_size as i32],
            &[(batch_size * hidden_size) as i32, hidden_size as i32, 1],
        )
        .unwrap();

        let rnn_desc = rnn_descriptor::<T, T>(
            algo,
            cell,
            bias,
            bidirectional,
            math,
            input_size,
            hidden_size,
            num_layers,
            dropout,
        )
        .unwrap();

        let weights_size = rnn_weight_space(rnn_desc).unwrap();

        Self {
            rnn_desc,
            h_desc,
            c_desc,
            weights_size,
            input_size,
            hidden_size,
            num_layers,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn get_workspace_reserve_size(
        &self,
        is_training: bool,
        x_desc: cudnnRNNDataDescriptor_t,
    ) -> RnnWorkspace {
        let handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();

        let fwd_mode = if is_training {
            cudnnForwardMode_t::CUDNN_FWD_MODE_TRAINING
        } else {
            cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE
        };

        let mut workspace_size: usize = 0;
        let mut reserve_size: usize = 0;

        let status = unsafe {
            cudnnGetRNNTempSpaceSizes(
                handle,
                self.rnn_desc,
                fwd_mode,
                x_desc,
                &mut workspace_size as *mut usize,
                &mut reserve_size as *mut usize,
            )
        };
        if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            panic!("Failed to get RNN temp space sizes: {:?}", status);
        }

        RnnWorkspace {
            workspace_size,
            reserve_size,
        }
    }
}

impl<T: 'static> Drop for RnnConfig<T> {
    fn drop(&mut self) {
        unsafe {
            zenu_cudnn_sys::cudnnDestroyRNNDescriptor(self.rnn_desc);
            zenu_cudnn_sys::cudnnDestroyTensorDescriptor(self.h_desc);
            zenu_cudnn_sys::cudnnDestroyTensorDescriptor(self.c_desc);
        }
    }
}

pub struct RNNExecutor<'a, T: 'static> {
    pub config: &'a RnnConfig<T>,
    pub x_desc: cudnnRNNDataDescriptor_t,
    pub y_desc: cudnnRNNDataDescriptor_t,
    pub workspace: RnnWorkspace,
    pub is_training: bool,
}

impl<'a, T: 'static + Clone + Copy> RNNExecutor<'a, T> {
    pub fn new(
        config: &'a RnnConfig<T>,
        seq_lengh: usize,
        batch_size: usize,
        seq_length_array: &[usize],
        layout: RNNDataLayout,
        fill_value: T,
        is_training: bool,
    ) -> Self {
        let seq_len_array = seq_length_array
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<i32>>();
        let x_desc = rnn_data_descriptor::<T>(
            seq_lengh as i32,
            batch_size as i32,
            config.input_size as i32,
            &seq_len_array,
            layout,
            fill_value,
        )
        .unwrap();

        let y_desc = rnn_data_descriptor::<T>(
            seq_lengh as i32,
            batch_size as i32,
            config.hidden_size as i32,
            &seq_len_array,
            layout,
            fill_value,
        )
        .unwrap();
        let workspace = config.get_workspace_reserve_size(is_training, x_desc);
        Self {
            config,
            x_desc,
            y_desc,
            workspace,
            is_training,
        }
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
        workspace: *mut T,
        reserve: *mut T,
    ) {
        rnn_fwd(
            self.config.rnn_desc,
            self.is_training,
            self.x_desc,
            x,
            self.y_desc,
            y,
            self.config.h_desc,
            hx,
            hy,
            self.config.c_desc,
            cx,
            cy,
            self.config.weights_size,
            weight,
            self.workspace.reserve_size,
            workspace,
            self.workspace.reserve_size,
            reserve,
        )
        .unwrap();
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
        workspace: *mut T,
        reserve: *mut T,
    ) {
        rnn_bkwd_data(
            self.config.rnn_desc,
            self.y_desc,
            y,
            dy,
            self.x_desc,
            dx,
            self.config.h_desc,
            hx,
            dhy,
            dhx,
            self.config.c_desc,
            cx,
            dcy,
            dcx,
            self.config.weights_size,
            weight,
            self.workspace.workspace_size,
            workspace,
            self.workspace.reserve_size,
            reserve,
        )
        .unwrap();
    }

    pub fn bkwd_weights(
        &self,
        x: *const T,
        hx: *const T,
        y: *const T,
        dweight: *mut T,
        workspace: *mut T,
        reserve: *mut T,
    ) {
        rnn_bkwd_weight(
            self.config.rnn_desc,
            self.x_desc,
            x,
            self.config.h_desc,
            hx,
            self.y_desc,
            y,
            self.config.weights_size,
            dweight,
            self.workspace.workspace_size,
            workspace,
            self.workspace.reserve_size,
            reserve,
        )
        .unwrap();
    }
}

impl<'a, T: 'static> Drop for RNNExecutor<'a, T> {
    fn drop(&mut self) {
        unsafe {
            zenu_cudnn_sys::cudnnDestroyRNNDataDescriptor(self.x_desc);
            zenu_cudnn_sys::cudnnDestroyRNNDataDescriptor(self.y_desc);
        }
    }
}

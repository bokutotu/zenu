use zenu_cudnn_sys::{
    cudnnDropoutDescriptor_t, cudnnForwardMode_t, cudnnGetRNNTempSpaceSizes,
    cudnnRNNDataDescriptor_t, cudnnRNNDescriptor_t, cudnnStatus_t, cudnnTensorDescriptor_t,
};

use crate::{cudnn::tensor_descriptor_nd, ZENU_CUDA_STATE};

use super::{
    helper::{
        rnn_bkwd_data, rnn_bkwd_weight, rnn_data_descriptor, rnn_descriptor, rnn_fwd,
        rnn_weight_params, rnn_weight_space,
    },
    RNNAlgo, RNNBias, RNNCell, RNNDataLayout, RNNMathType,
};

#[expect(clippy::module_name_repetitions)]
pub struct RNNDescriptor<T: 'static + Copy> {
    rnn_desc: cudnnRNNDescriptor_t,
    h_desc: cudnnTensorDescriptor_t,
    c_desc: cudnnTensorDescriptor_t,
    weights_size: usize,
    input_size: usize,
    hidden_size: usize,
    batch_size: usize,
    num_layers: usize,
    cell: RNNCell,
    bidirectional: bool,
    context: Option<RNNContext>,
    _marker: std::marker::PhantomData<T>,
}

pub struct RnnWorkspace {
    pub workspace_size: usize,
    pub reserve_size: usize,
}

pub struct RNNDescPtr {
    pub desc: cudnnTensorDescriptor_t,
    pub ptr: *mut std::ffi::c_void,
}

pub struct RNNParams {
    pub input_weight: RNNDescPtr,
    pub hidden_weight: RNNDescPtr,
    pub input_bias: RNNDescPtr,
    pub hidden_bias: RNNDescPtr,
}

impl<T: 'static + Copy> RNNDescriptor<T> {
    #[expect(clippy::too_many_arguments)]
    #[must_use]
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
        let (h_desc, c_desc) = {
            let num_layers = i32::try_from(num_layers).unwrap();
            let batch_size = i32::try_from(batch_size).unwrap();
            let hidden_size = i32::try_from(hidden_size).unwrap();

            let h_num_layers = num_layers * if bidirectional { 2 } else { 1 };
            let h_desc = tensor_descriptor_nd::<T>(
                &[h_num_layers, batch_size, hidden_size],
                &[batch_size * hidden_size, hidden_size, 1],
            )
            .unwrap();

            let c_desc = tensor_descriptor_nd::<T>(
                &[h_num_layers, batch_size, hidden_size],
                &[batch_size * hidden_size, hidden_size, 1],
            )
            .unwrap();
            (h_desc, c_desc)
        };

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
            batch_size,
            cell,
            bidirectional,
            context: None,
            _marker: std::marker::PhantomData,
        }
    }

    #[expect(clippy::not_unsafe_ptr_arg_deref)]
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
                std::ptr::from_mut(&mut workspace_size),
                std::ptr::from_mut(&mut reserve_size),
            )
        };

        let error_message = format!("Failed to get RNN temp space sizes: {status:?}");
        assert_eq!(
            status,
            cudnnStatus_t::CUDNN_STATUS_SUCCESS,
            "{error_message}"
        );

        RnnWorkspace {
            workspace_size,
            reserve_size,
        }
    }

    #[must_use]
    pub fn get_input_size(&self) -> usize {
        self.input_size
    }

    #[must_use]
    pub fn get_hidden_size(&self) -> usize {
        self.hidden_size
    }

    #[must_use]
    pub fn get_num_layers(&self) -> usize {
        self.num_layers
    }

    #[must_use]
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    #[must_use]
    pub fn get_is_bidirectional(&self) -> bool {
        self.bidirectional
    }

    pub fn get_rnn_params(&self, weight_ptr: *mut T) -> Vec<RNNParams> {
        assert!(
            !(self.cell != RNNCell::RNNRelu && self.cell != RNNCell::RNNTanh),
            "Only RNN cell is supported"
        );

        let mut params = Vec::new();

        let num_layers = self.num_layers * if self.bidirectional { 2 } else { 1 };

        for layer_idx in 0..num_layers {
            let input_params =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 0)
                    .unwrap();
            let hidden_params =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 1)
                    .unwrap();

            params.push(RNNParams {
                input_weight: RNNDescPtr {
                    desc: input_params.weight_desc,
                    ptr: input_params.weight,
                },
                hidden_weight: RNNDescPtr {
                    desc: hidden_params.weight_desc,
                    ptr: hidden_params.weight,
                },
                input_bias: RNNDescPtr {
                    desc: input_params.bias_desc,
                    ptr: input_params.bias,
                },
                hidden_bias: RNNDescPtr {
                    desc: hidden_params.bias_desc,
                    ptr: hidden_params.bias,
                },
            });
        }

        params
    }

    pub fn set_input_size(
        &mut self,
        seq_length: usize,
        seq_length_array: &[usize],
        layout: RNNDataLayout,
        is_training: bool,
        fill_value: T,
    ) {
        let seq_length = i32::try_from(seq_length).unwrap();
        let batch_size = i32::try_from(self.batch_size).unwrap();
        let input_size = i32::try_from(self.input_size).unwrap();
        let seq_len_array = seq_length_array
            .iter()
            .map(|&x| i32::try_from(x).unwrap())
            .collect::<Vec<i32>>();
        let x_desc = rnn_data_descriptor::<T>(
            seq_length,
            batch_size,
            input_size,
            &seq_len_array,
            layout,
            fill_value,
        )
        .unwrap();

        let hidden_size = self.hidden_size * if self.bidirectional { 2 } else { 1 };
        let hidden_size = i32::try_from(hidden_size).unwrap();

        let y_desc = rnn_data_descriptor::<T>(
            seq_length,
            batch_size,
            hidden_size,
            &seq_len_array,
            layout,
            fill_value,
        )
        .unwrap();
        let workspace = self.get_workspace_reserve_size(is_training, x_desc);
        self.context = Some(RNNContext {
            x_desc,
            y_desc,
            workspace,
            is_training,
        });
    }

    #[expect(clippy::too_many_arguments)]
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
            self.rnn_desc,
            self.context.as_ref().unwrap().is_training,
            self.context.as_ref().unwrap().x_desc,
            x,
            self.context.as_ref().unwrap().y_desc,
            y,
            self.h_desc,
            hx,
            hy,
            self.c_desc,
            cx,
            cy,
            self.weights_size,
            weight,
            self.context.as_ref().unwrap().get_workspace_size(),
            workspace,
            self.context.as_ref().unwrap().get_reserve_size(),
            reserve,
        )
        .unwrap();
    }

    #[expect(clippy::too_many_arguments, clippy::similar_names)]
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
            self.rnn_desc,
            self.context.as_ref().unwrap().y_desc,
            y,
            dy,
            self.context.as_ref().unwrap().x_desc,
            dx,
            self.h_desc,
            hx,
            dhy,
            dhx,
            self.c_desc,
            cx,
            dcy,
            dcx,
            self.weights_size,
            weight,
            self.context.as_ref().unwrap().get_workspace_size(),
            workspace,
            self.context.as_ref().unwrap().get_reserve_size(),
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
            self.rnn_desc,
            self.context.as_ref().unwrap().x_desc,
            x,
            self.h_desc,
            hx,
            self.context.as_ref().unwrap().y_desc,
            y,
            self.weights_size,
            dweight,
            self.context.as_ref().unwrap().get_workspace_size(),
            workspace,
            self.context.as_ref().unwrap().get_reserve_size(),
            reserve,
        )
        .unwrap();
    }

    #[must_use]
    pub fn get_weights_size(&self) -> usize {
        self.weights_size
    }

    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn get_workspace_size(&self) -> usize {
        self.context.as_ref().unwrap().get_workspace_size()
    }

    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn get_reserve_size(&self) -> usize {
        self.context.as_ref().unwrap().get_reserve_size()
    }

    #[must_use]
    pub fn context_is_none(&self) -> bool {
        self.context.is_none()
    }
}

impl<T: 'static + Copy> Drop for RNNDescriptor<T> {
    fn drop(&mut self) {
        unsafe {
            zenu_cudnn_sys::cudnnDestroyRNNDescriptor(self.rnn_desc);
            zenu_cudnn_sys::cudnnDestroyTensorDescriptor(self.h_desc);
            zenu_cudnn_sys::cudnnDestroyTensorDescriptor(self.c_desc);
        }
    }
}

pub struct RNNContext {
    pub x_desc: cudnnRNNDataDescriptor_t,
    pub y_desc: cudnnRNNDataDescriptor_t,
    pub workspace: RnnWorkspace,
    pub is_training: bool,
}

impl RNNContext {
    #[must_use]
    pub fn get_reserve_size(&self) -> usize {
        self.workspace.reserve_size
    }

    #[must_use]
    pub fn get_workspace_size(&self) -> usize {
        self.workspace.workspace_size
    }
}

impl Drop for RNNContext {
    fn drop(&mut self) {
        unsafe {
            zenu_cudnn_sys::cudnnDestroyRNNDataDescriptor(self.x_desc);
            zenu_cudnn_sys::cudnnDestroyRNNDataDescriptor(self.y_desc);
        }
    }
}

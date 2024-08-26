use zenu_cudnn_sys::{
    cudnnDropoutDescriptor_t, cudnnForwardMode_t, cudnnGetRNNTempSpaceSizes,
    cudnnRNNDataDescriptor_t, cudnnRNNDescriptor_t, cudnnTensorDescriptor_t,
};

use crate::{cudnn::tensor_descriptor_nd, ZENU_CUDA_STATE};

use super::{
    helper::{
        rnn_bkwd_data, rnn_bkwd_weight, rnn_data_descriptor, rnn_descriptor, rnn_fwd,
        rnn_weight_params, rnn_weight_space,
    },
    RNNAlgo, RNNBias, RNNCell, RNNDataLayout, RNNMathType,
};

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

pub struct LstmParams {
    pub input_gate_x: RNNDescPtr,
    pub input_gate_h: RNNDescPtr,
    pub forget_gate_x: RNNDescPtr,
    pub forget_gate_h: RNNDescPtr,
    pub cell_x: RNNDescPtr,
    pub cell_h: RNNDescPtr,
    pub output_gate_x: RNNDescPtr,
    pub output_gate_h: RNNDescPtr,
}

pub struct RNNParams {
    pub input_weight: RNNDescPtr,
    pub hidden_weight: RNNDescPtr,
    pub input_bias: RNNDescPtr,
    pub hidden_bias: RNNDescPtr,
}

pub struct GRUParams {
    pub reset_gate_x: RNNDescPtr,
    pub reset_gate_h: RNNDescPtr,
    pub update_gate_x: RNNDescPtr,
    pub update_gate_h: RNNDescPtr,
    pub cell_x: RNNDescPtr,
    pub cell_h: RNNDescPtr,
}

impl<T: 'static + Copy> RNNDescriptor<T> {
    #[allow(clippy::too_many_arguments)]
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
            batch_size,
            cell,
            bidirectional,
            context: None,
            _marker: std::marker::PhantomData,
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
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

    pub fn get_rnn_params(&self, weight_ptr: *mut T) -> Vec<RNNParams> {
        if self.cell != RNNCell::RNNRelu && self.cell != RNNCell::RNNTanh {
            panic!("Only RNN cell is supported");
        }
        let mut params = Vec::new();

        for layer_idx in 0..self.num_layers {
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

    pub fn get_lstm_params(&self, weight_ptr: *mut T) -> Vec<LstmParams> {
        if self.cell != RNNCell::LSTM {
            panic!("Only LSTM cell is supported");
        }
        let mut params = Vec::new();

        for layer_idx in 0..self.num_layers {
            let input_gate_x =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 0)
                    .unwrap();
            let input_gate_h =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 1)
                    .unwrap();
            let forget_gate_x =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 2)
                    .unwrap();
            let forget_gate_h =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 3)
                    .unwrap();
            let cell_x =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 4)
                    .unwrap();
            let cell_h =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 5)
                    .unwrap();
            let output_gate_x =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 6)
                    .unwrap();
            let output_gate_h =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 7)
                    .unwrap();

            params.push(LstmParams {
                input_gate_x: RNNDescPtr {
                    desc: input_gate_x.weight_desc,
                    ptr: input_gate_x.weight,
                },
                input_gate_h: RNNDescPtr {
                    desc: input_gate_h.weight_desc,
                    ptr: input_gate_h.weight,
                },
                forget_gate_x: RNNDescPtr {
                    desc: forget_gate_x.weight_desc,
                    ptr: forget_gate_x.weight,
                },
                forget_gate_h: RNNDescPtr {
                    desc: forget_gate_h.weight_desc,
                    ptr: forget_gate_h.weight,
                },
                cell_x: RNNDescPtr {
                    desc: cell_x.weight_desc,
                    ptr: cell_x.weight,
                },
                cell_h: RNNDescPtr {
                    desc: cell_h.weight_desc,
                    ptr: cell_h.weight,
                },
                output_gate_x: RNNDescPtr {
                    desc: output_gate_x.weight_desc,
                    ptr: output_gate_x.weight,
                },
                output_gate_h: RNNDescPtr {
                    desc: output_gate_h.weight_desc,
                    ptr: output_gate_h.weight,
                },
            });
        }

        params
    }

    pub fn get_gru_params(&self, weight_ptr: *mut T) -> Vec<GRUParams> {
        if self.cell != RNNCell::GRU {
            panic!("Only GRU cell is supported");
        }
        let mut params = Vec::new();

        for layer_idx in 0..self.num_layers {
            let reset_gate_x =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 0)
                    .unwrap();
            let reset_gate_h =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 1)
                    .unwrap();
            let update_gate_x =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 2)
                    .unwrap();
            let update_gate_h =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 3)
                    .unwrap();
            let cell_x =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 4)
                    .unwrap();
            let cell_h =
                rnn_weight_params(self.rnn_desc, layer_idx, self.weights_size, weight_ptr, 5)
                    .unwrap();

            params.push(GRUParams {
                reset_gate_x: RNNDescPtr {
                    desc: reset_gate_x.weight_desc,
                    ptr: reset_gate_x.weight,
                },
                reset_gate_h: RNNDescPtr {
                    desc: reset_gate_h.weight_desc,
                    ptr: reset_gate_h.weight,
                },
                update_gate_x: RNNDescPtr {
                    desc: update_gate_x.weight_desc,
                    ptr: update_gate_x.weight,
                },
                update_gate_h: RNNDescPtr {
                    desc: update_gate_h.weight_desc,
                    ptr: update_gate_h.weight,
                },
                cell_x: RNNDescPtr {
                    desc: cell_x.weight_desc,
                    ptr: cell_x.weight,
                },
                cell_h: RNNDescPtr {
                    desc: cell_h.weight_desc,
                    ptr: cell_h.weight,
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
        let seq_len_array = seq_length_array
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<i32>>();
        let x_desc = rnn_data_descriptor::<T>(
            seq_length as i32,
            self.batch_size as i32,
            self.input_size as i32,
            &seq_len_array,
            layout,
            fill_value,
        )
        .unwrap();

        let y_desc = rnn_data_descriptor::<T>(
            seq_length as i32,
            self.batch_size as i32,
            self.hidden_size as i32,
            &seq_len_array,
            layout,
            fill_value,
        )
        .unwrap();
        let workspace = self.get_workspace_reserve_size(is_training, x_desc);
        self.context = Some(RNNContext::<T> {
            x_desc,
            y_desc,
            workspace,
            is_training,
        });
    }

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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

    pub fn get_weights_size(&self) -> usize {
        self.weights_size
    }

    pub fn get_workspace_size(&self) -> usize {
        self.context.as_ref().unwrap().get_workspace_size()
    }

    pub fn get_reserve_size(&self) -> usize {
        self.context.as_ref().unwrap().get_reserve_size()
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
    // pub fn new(
    //     seq_lengh: usize,
    //     seq_length_array: &[usize],
    //     layout: RNNDataLayout,
    //     fill_value: T,
    //     is_training: bool,
    // ) -> Self {
    //     let seq_len_array = seq_length_array
    //         .iter()
    //         .map(|&x| x as i32)
    //         .collect::<Vec<i32>>();
    //     let x_desc = rnn_data_descriptor::<T>(
    //         seq_lengh as i32,
    //         config.batch_size as i32,
    //         config.input_size as i32,
    //         &seq_len_array,
    //         layout,
    //         fill_value,
    //     )
    //     .unwrap();
    //
    //     let y_desc = rnn_data_descriptor::<T>(
    //         seq_lengh as i32,
    //         config.batch_size as i32,
    //         config.hidden_size as i32,
    //         &seq_len_array,
    //         layout,
    //         fill_value,
    //     )
    //     .unwrap();
    //     let workspace = config.get_workspace_reserve_size(is_training, x_desc);
    //     Self {
    //         config,
    //         x_desc,
    //         y_desc,
    //         workspace,
    //         is_training,
    //     }
    // }

    // #[allow(clippy::too_many_arguments)]
    // pub fn fwd(
    //     &self,
    //     x: *const T,
    //     y: *mut T,
    //     hx: *const T,
    //     hy: *mut T,
    //     cx: *const T,
    //     cy: *mut T,
    //     weight: *mut T,
    //     workspace: *mut T,
    //     reserve: *mut T,
    // ) {
    //     rnn_fwd(
    //         self.config.rnn_desc,
    //         self.is_training,
    //         self.x_desc,
    //         x,
    //         self.y_desc,
    //         y,
    //         self.config.h_desc,
    //         hx,
    //         hy,
    //         self.config.c_desc,
    //         cx,
    //         cy,
    //         self.config.weights_size,
    //         weight,
    //         self.workspace.workspace_size,
    //         workspace,
    //         self.workspace.reserve_size,
    //         reserve,
    //     )
    //     .unwrap();
    // }
    //
    // #[allow(clippy::too_many_arguments)]
    // pub fn bkwd_data(
    //     &self,
    //     y: *const T,
    //     dy: *const T,
    //     dx: *mut T,
    //     hx: *const T,
    //     dhy: *const T,
    //     dhx: *mut T,
    //     cx: *const T,
    //     dcy: *const T,
    //     dcx: *mut T,
    //     weight: *const T,
    //     workspace: *mut T,
    //     reserve: *mut T,
    // ) {
    //     rnn_bkwd_data(
    //         self.config.rnn_desc,
    //         self.y_desc,
    //         y,
    //         dy,
    //         self.x_desc,
    //         dx,
    //         self.config.h_desc,
    //         hx,
    //         dhy,
    //         dhx,
    //         self.config.c_desc,
    //         cx,
    //         dcy,
    //         dcx,
    //         self.config.weights_size,
    //         weight,
    //         self.workspace.workspace_size,
    //         workspace,
    //         self.workspace.reserve_size,
    //         reserve,
    //     )
    //     .unwrap();
    // }
    //
    // pub fn bkwd_weights(
    //     &self,
    //     x: *const T,
    //     hx: *const T,
    //     y: *const T,
    //     dweight: *mut T,
    //     workspace: *mut T,
    //     reserve: *mut T,
    // ) {
    //     rnn_bkwd_weight(
    //         self.config.rnn_desc,
    //         self.x_desc,
    //         x,
    //         self.config.h_desc,
    //         hx,
    //         self.y_desc,
    //         y,
    //         self.config.weights_size,
    //         dweight,
    //         self.workspace.workspace_size,
    //         workspace,
    //         self.workspace.reserve_size,
    //         reserve,
    //     )
    //     .unwrap();
    // }

    pub fn get_reserve_size(&self) -> usize {
        self.workspace.reserve_size
    }

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

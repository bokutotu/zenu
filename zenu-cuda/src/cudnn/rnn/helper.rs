use std::any::TypeId;

use zenu_cudnn_sys::{
    cudnnCreateTensorDescriptor, cudnnDataType_t, cudnnDirectionMode_t, cudnnDropoutDescriptor_t,
    cudnnForwardMode_t, cudnnGetRNNWeightParams, cudnnGetRNNWeightSpaceSize, cudnnMathType_t,
    cudnnRNNAlgo_t, cudnnRNNBackwardData_v8, cudnnRNNBackwardWeights_v8, cudnnRNNBiasMode_t,
    cudnnRNNDataDescriptor_t, cudnnRNNDataLayout_t, cudnnRNNDescriptor_t, cudnnRNNForward,
    cudnnRNNInputMode_t, cudnnRNNMode_t, cudnnSetRNNDataDescriptor, cudnnSetRNNDescriptor_v8,
    cudnnStatus_t, cudnnTensorDescriptor_t, cudnnWgradMode_t, CUDNN_RNN_PADDED_IO_DISABLED,
};

use crate::ZENU_CUDA_STATE;

use super::super::error::ZenuCudnnError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNAlgo {
    Standard,
    PersistStatic,
    PersistDynamic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNCell {
    LSTM,
    GRU,
    RNNRelu,
    RNNTanh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNBias {
    NoBias,
    SingleInpBias,
    SingleRecBias,
    DoubleBias,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNMathType {
    Default,
    TensorOp,
    TensorOpAllowConversion,
}

#[expect(clippy::too_many_arguments)]
pub fn rnn_descriptor<Data: 'static, Math: 'static>(
    algo: RNNAlgo,
    cell: RNNCell,
    bias: RNNBias,
    birectional: bool,
    math_type: RNNMathType,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    dropout: Option<cudnnDropoutDescriptor_t>,
) -> Result<cudnnRNNDescriptor_t, ZenuCudnnError> {
    let mut rnn_desc: cudnnRNNDescriptor_t = std::ptr::null_mut();
    let status = unsafe { zenu_cudnn_sys::cudnnCreateRNNDescriptor(&mut rnn_desc) };
    if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }

    let algo = match algo {
        RNNAlgo::Standard => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD,
        RNNAlgo::PersistStatic => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_STATIC,
        RNNAlgo::PersistDynamic => cudnnRNNAlgo_t::CUDNN_RNN_ALGO_PERSIST_DYNAMIC,
    };

    let cell = match cell {
        RNNCell::LSTM => cudnnRNNMode_t::CUDNN_LSTM,
        RNNCell::GRU => cudnnRNNMode_t::CUDNN_GRU,
        RNNCell::RNNRelu => cudnnRNNMode_t::CUDNN_RNN_RELU,
        RNNCell::RNNTanh => cudnnRNNMode_t::CUDNN_RNN_TANH,
    };

    let bias = match bias {
        RNNBias::NoBias => cudnnRNNBiasMode_t::CUDNN_RNN_NO_BIAS,
        RNNBias::SingleInpBias => cudnnRNNBiasMode_t::CUDNN_RNN_SINGLE_INP_BIAS,
        RNNBias::SingleRecBias => cudnnRNNBiasMode_t::CUDNN_RNN_SINGLE_REC_BIAS,
        RNNBias::DoubleBias => cudnnRNNBiasMode_t::CUDNN_RNN_DOUBLE_BIAS,
    };

    let math_type = match math_type {
        RNNMathType::Default => cudnnMathType_t::CUDNN_DEFAULT_MATH,
        RNNMathType::TensorOp => cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
        RNNMathType::TensorOpAllowConversion => {
            cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
        }
    };

    let dire = if birectional {
        cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL
    } else {
        cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL
    };

    let data_type = if TypeId::of::<Data>() == TypeId::of::<f32>() {
        cudnnDataType_t::CUDNN_DATA_FLOAT
    } else if TypeId::of::<Data>() == TypeId::of::<f64>() {
        cudnnDataType_t::CUDNN_DATA_DOUBLE
    } else {
        return Err(ZenuCudnnError::Other);
    };

    let math_prec = if TypeId::of::<Math>() == TypeId::of::<f32>() {
        cudnnDataType_t::CUDNN_DATA_FLOAT
    } else if TypeId::of::<Math>() == TypeId::of::<f64>() {
        cudnnDataType_t::CUDNN_DATA_DOUBLE
    } else {
        return Err(ZenuCudnnError::Other);
    };

    let dropout = if let Some(dropout) = dropout {
        dropout
    } else {
        std::ptr::null_mut()
    };

    let status = unsafe {
        cudnnSetRNNDescriptor_v8(
            rnn_desc,
            algo,
            cell,
            bias,
            dire,
            cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
            data_type,
            math_prec,
            math_type,
            i32::try_from(input_size).unwrap(),
            i32::try_from(hidden_size).unwrap(),
            i32::try_from(hidden_size).unwrap(),
            i32::try_from(num_layers).unwrap(),
            dropout,
            CUDNN_RNN_PADDED_IO_DISABLED,
        )
    };

    if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }

    Ok(rnn_desc)
}

pub fn rnn_weight_space(rnn_desc: cudnnRNNDescriptor_t) -> Result<usize, ZenuCudnnError> {
    let mut size: usize = 0;

    let handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();

    let status =
        unsafe { cudnnGetRNNWeightSpaceSize(handle, rnn_desc, std::ptr::from_mut(&mut size)) };
    if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }

    Ok(size)
}

#[expect(clippy::too_many_arguments)]
pub fn rnn_fwd<T: 'static>(
    rnn_desc: cudnnRNNDescriptor_t,
    is_training: bool,
    x_desc: cudnnRNNDataDescriptor_t,
    x: *const T,
    y_desc: cudnnRNNDataDescriptor_t,
    y: *mut T,
    h_desc: cudnnTensorDescriptor_t,
    hx: *const T,
    hy: *mut T,
    c_desc: cudnnTensorDescriptor_t,
    cx: *const T,
    cy: *mut T,
    weight_size: usize,
    weight: *const T,
    workspace_size: usize,
    workspace: *mut T,
    reserve_size: usize,
    reserve: *mut T,
) -> Result<(), ZenuCudnnError> {
    let handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();

    let fwd_mode = if is_training {
        cudnnForwardMode_t::CUDNN_FWD_MODE_TRAINING
    } else {
        cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE
    };

    let status = unsafe {
        cudnnRNNForward(
            handle,
            rnn_desc,
            fwd_mode,
            std::ptr::null_mut(),
            x_desc,
            x.cast::<::libc::c_void>(),
            y_desc,
            y.cast::<::libc::c_void>(),
            h_desc,
            hx.cast::<::libc::c_void>(),
            hy.cast::<::libc::c_void>(),
            c_desc,
            cx.cast::<::libc::c_void>(),
            cy.cast::<::libc::c_void>(),
            weight_size,
            weight.cast::<::libc::c_void>(),
            workspace_size,
            workspace.cast::<::libc::c_void>(),
            reserve_size,
            reserve.cast::<::libc::c_void>(),
        )
    };

    if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }

    Ok(())
}

#[expect(clippy::too_many_arguments, clippy::similar_names)]
pub fn rnn_bkwd_data<T: 'static>(
    rnn_desc: cudnnRNNDescriptor_t,
    y_desc: cudnnRNNDataDescriptor_t,
    y: *const T,
    dy: *const T,
    x_desc: cudnnRNNDataDescriptor_t,
    dx: *mut T,
    h_desc: cudnnTensorDescriptor_t,
    hx: *const T,
    dhy: *const T,
    dhx: *mut T,
    c_desc: cudnnTensorDescriptor_t,
    cx: *const T,
    dcy: *const T,
    dcx: *mut T,
    weight_size: usize,
    weight: *const T,
    workspace_size: usize,
    workspace: *mut T,
    reserve_size: usize,
    reserve: *mut T,
) -> Result<(), ZenuCudnnError> {
    let handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();

    let status = unsafe {
        cudnnRNNBackwardData_v8(
            handle,
            rnn_desc,
            std::ptr::null_mut(),
            y_desc,
            y.cast::<::libc::c_void>(),
            dy.cast::<::libc::c_void>(),
            x_desc,
            dx.cast::<::libc::c_void>(),
            h_desc,
            hx.cast::<::libc::c_void>(),
            dhy.cast::<::libc::c_void>(),
            dhx.cast::<::libc::c_void>(),
            c_desc,
            cx.cast::<::libc::c_void>(),
            dcy.cast::<::libc::c_void>(),
            dcx.cast::<::libc::c_void>(),
            weight_size,
            weight.cast::<::libc::c_void>(),
            workspace_size,
            workspace.cast::<::libc::c_void>(),
            reserve_size,
            reserve.cast::<::libc::c_void>(),
        )
    };

    if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }

    Ok(())
}

#[expect(clippy::too_many_arguments)]
pub fn rnn_bkwd_weight<T: 'static>(
    rnn_desc: cudnnRNNDescriptor_t,
    x_desc: cudnnRNNDataDescriptor_t,
    x: *const T,
    h_desc: cudnnTensorDescriptor_t,
    hx: *const T,
    y_desc: cudnnRNNDataDescriptor_t,
    y: *const T,
    weight_size: usize,
    dweight: *mut T,
    workspace_size: usize,
    workspace: *mut T,
    reserve_size: usize,
    reserve: *mut T,
) -> Result<(), ZenuCudnnError> {
    let handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();

    let status = unsafe {
        cudnnRNNBackwardWeights_v8(
            handle,
            rnn_desc,
            cudnnWgradMode_t::CUDNN_WGRAD_MODE_ADD,
            std::ptr::null_mut(),
            x_desc,
            x.cast::<::libc::c_void>(),
            h_desc,
            hx.cast::<::libc::c_void>(),
            y_desc,
            y.cast::<::libc::c_void>(),
            weight_size,
            dweight.cast::<::libc::c_void>(),
            workspace_size,
            workspace.cast::<::libc::c_void>(),
            reserve_size,
            reserve.cast::<::libc::c_void>(),
        )
    };

    if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub enum RNNDataLayout {
    SeqMajorUnpacked,
    SeqMajorPacked,
    BatchMajorUnpacked,
}

pub fn rnn_data_descriptor<T: 'static>(
    max_seq_len: i32,
    batch_size: i32,
    vector_size: i32,
    seq_len_array: &[i32],
    layout: RNNDataLayout,
    mut fill_value: T,
) -> Result<cudnnRNNDataDescriptor_t, ZenuCudnnError> {
    let data_type = if TypeId::of::<T>() == TypeId::of::<f32>() {
        cudnnDataType_t::CUDNN_DATA_FLOAT
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        cudnnDataType_t::CUDNN_DATA_DOUBLE
    } else {
        return Err(ZenuCudnnError::Other);
    };

    let layout = match layout {
        RNNDataLayout::SeqMajorUnpacked => {
            cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED
        }
        RNNDataLayout::SeqMajorPacked => {
            cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED
        }
        RNNDataLayout::BatchMajorUnpacked => {
            cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
        }
    };

    assert!(
        seq_len_array.len() == usize::try_from(batch_size).unwrap(),
        "seq_len_array length must be equal to batch_size"
    );

    for seq_i in seq_len_array {
        assert!(
            *seq_i <= max_seq_len,
            "seq_len_array contains a value greater than max_req_len"
        );
    }

    let mut tensor: cudnnRNNDataDescriptor_t = std::ptr::null_mut();
    let status = unsafe {
        zenu_cudnn_sys::cudnnCreateRNNDataDescriptor(
            std::ptr::from_mut::<cudnnRNNDataDescriptor_t>(&mut tensor),
        )
    };
    if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }

    let status = unsafe {
        cudnnSetRNNDataDescriptor(
            tensor,
            data_type,
            layout,
            max_seq_len as ::libc::c_int,
            batch_size as ::libc::c_int,
            vector_size as ::libc::c_int,
            seq_len_array.as_ptr(),
            std::ptr::from_mut::<T>(&mut fill_value).cast::<::libc::c_void>(),
        )
    };
    if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }
    Ok(tensor)
}

pub struct RNNWeightParams {
    pub weight_desc: cudnnTensorDescriptor_t,
    pub weight: *mut ::libc::c_void,
    pub bias_desc: cudnnTensorDescriptor_t,
    pub bias: *mut ::libc::c_void,
}

pub fn rnn_weight_params<T: 'static>(
    rnn_desc: cudnnRNNDescriptor_t,
    pseudo_layer: usize,
    weight_size: usize,
    weight: *mut T,
    leyer_id: usize,
) -> Result<RNNWeightParams, ZenuCudnnError> {
    let handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();

    let mut weight_desc: cudnnTensorDescriptor_t = std::ptr::null_mut();
    let mut bias_desc: cudnnTensorDescriptor_t = std::ptr::null_mut();

    let status = unsafe { cudnnCreateTensorDescriptor(std::ptr::from_mut(&mut weight_desc)) };
    if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }

    let status = unsafe { cudnnCreateTensorDescriptor(std::ptr::from_mut(&mut bias_desc)) };
    if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }

    let mut m_addr = std::ptr::null_mut();
    let m_addr_addr = &mut m_addr as *mut *mut ::libc::c_void;

    let mut b_addr = std::ptr::null_mut();
    let b_addr_addr = &mut b_addr as *mut *mut ::libc::c_void;

    let status = unsafe {
        cudnnGetRNNWeightParams(
            handle,
            rnn_desc,
            i32::try_from(pseudo_layer).unwrap(),
            weight_size,
            // weight as *mut ::libc::c_void,
            weight.cast::<::libc::c_void>(),
            i32::try_from(leyer_id).unwrap(),
            weight_desc,
            m_addr_addr,
            bias_desc,
            b_addr_addr,
        )
    };
    if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }

    Ok(RNNWeightParams {
        weight_desc,
        weight: m_addr,
        bias_desc,
        bias: b_addr,
    })
}

#[cfg(test)]
mod rnn {
    use zenu_cudnn_sys::{
        cudnnForwardMode_t, cudnnGetRNNTempSpaceSizes, cudnnGetRNNWeightSpaceSize,
        cudnnRNNDataDescriptor_t, cudnnRNNDescriptor_t,
    };

    use crate::{
        cudnn::{error::ZenuCudnnError, tensor_descriptor_nd},
        runtime::cuda_malloc,
        ZENU_CUDA_STATE,
    };

    use super::{
        rnn_data_descriptor, rnn_descriptor, rnn_fwd, RNNAlgo, RNNBias, RNNCell, RNNDataLayout,
        RNNMathType,
    };

    pub struct RNNBytes {
        weights: usize,
        workspace: usize,
        reserve: usize,
    }

    impl RNNBytes {
        pub fn new(
            rnn_desc: cudnnRNNDescriptor_t,
            is_training: bool,
            x_desc: cudnnRNNDataDescriptor_t,
        ) -> Self {
            let weights_size = Self::rnn_weight_space(rnn_desc).unwrap();
            let (workspace_size, reserve_size) =
                Self::rnn_tmp_space(rnn_desc, is_training, x_desc).unwrap();
            Self {
                weights: weights_size,
                workspace: workspace_size,
                reserve: reserve_size,
            }
        }

        pub fn weights_size(&self) -> usize {
            self.weights
        }

        pub fn workspace_size(&self) -> usize {
            self.workspace
        }

        pub fn reserve_size(&self) -> usize {
            self.reserve
        }

        fn rnn_weight_space(rnn_desc: cudnnRNNDescriptor_t) -> Result<usize, ZenuCudnnError> {
            let mut size: usize = 0;

            let handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();

            let status = unsafe {
                cudnnGetRNNWeightSpaceSize(handle, rnn_desc, std::ptr::from_mut(&mut size))
            };
            if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(ZenuCudnnError::from(status));
            }

            Ok(size)
        }

        fn rnn_tmp_space(
            rnn_desc: cudnnRNNDescriptor_t,
            is_training: bool,
            x_desc: cudnnRNNDataDescriptor_t,
        ) -> Result<(usize, usize), ZenuCudnnError> {
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
                    rnn_desc,
                    fwd_mode,
                    x_desc,
                    std::ptr::from_mut(&mut workspace_size),
                    std::ptr::from_mut(&mut reserve_size),
                )
            };
            if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(ZenuCudnnError::from(status));
            }

            Ok((workspace_size, reserve_size))
        }
    }

    #[test]
    fn rnn_run_test() {
        let input_size = 64;
        let hidden_size = 128;
        let num_layers = 2;
        let seq_len = 20;
        let batch_size = 64;
        let rnn_desc = rnn_descriptor::<f32, f32>(
            RNNAlgo::Standard,
            RNNCell::RNNRelu,
            RNNBias::NoBias,
            false,
            RNNMathType::Default,
            input_size,
            hidden_size,
            num_layers,
            None,
        )
        .unwrap();

        let seq_len_array = vec![seq_len; batch_size];

        let x_data_desc = rnn_data_descriptor::<f32>(
            seq_len,
            i32::try_from(batch_size).unwrap(),
            i32::try_from(input_size).unwrap(),
            &seq_len_array,
            RNNDataLayout::SeqMajorUnpacked,
            0.,
        )
        .unwrap();

        let y_data_desc = rnn_data_descriptor::<f32>(
            seq_len,
            i32::try_from(batch_size).unwrap(),
            i32::try_from(hidden_size).unwrap(),
            &seq_len_array,
            RNNDataLayout::SeqMajorUnpacked,
            0.,
        )
        .unwrap();

        let x_data =
            cuda_malloc::<f32>(usize::try_from(seq_len).unwrap() * batch_size * input_size)
                .unwrap();

        let y_data =
            cuda_malloc::<f32>(usize::try_from(seq_len).unwrap() * batch_size * hidden_size)
                .unwrap();

        let bytes = RNNBytes::new(rnn_desc, true, x_data_desc);

        let workspace = cuda_malloc::<f32>(bytes.workspace_size() / size_of::<f32>()).unwrap();
        let reserve = cuda_malloc::<f32>(bytes.reserve_size() / size_of::<f32>()).unwrap();
        let weight = cuda_malloc::<f32>(bytes.weights_size() / size_of::<f32>()).unwrap();

        let hx = cuda_malloc::<f32>(256).unwrap();
        let hy = cuda_malloc::<f32>(256).unwrap();
        let cx = cuda_malloc::<f32>(256).unwrap();
        let cy = cuda_malloc::<f32>(256).unwrap();

        let num_layers = i32::try_from(num_layers).unwrap();
        let batch_size = i32::try_from(batch_size).unwrap();
        let hidden_size = i32::try_from(hidden_size).unwrap();

        let h_desc = tensor_descriptor_nd::<f32>(
            &[num_layers, batch_size, hidden_size],
            &[batch_size * hidden_size, hidden_size, 1],
        )
        .unwrap();

        let c_desc = tensor_descriptor_nd::<f32>(
            &[num_layers, batch_size, hidden_size],
            &[(batch_size * hidden_size), hidden_size, 1],
        )
        .unwrap();

        rnn_fwd(
            rnn_desc,
            true,
            x_data_desc,
            x_data,
            y_data_desc,
            y_data,
            h_desc,
            hx,
            hy,
            c_desc,
            cx,
            cy,
            bytes.weights_size(),
            weight,
            bytes.workspace_size(),
            workspace,
            bytes.reserve_size(),
            reserve,
        )
        .unwrap();
    }
}

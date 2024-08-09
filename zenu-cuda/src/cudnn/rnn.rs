use std::any::TypeId;

use zenu_cudnn_sys::{
    cudnnDataType_t, cudnnDirectionMode_t, cudnnDropoutDescriptor_t, cudnnMathType_t,
    cudnnRNNAlgo_t, cudnnRNNBiasMode_t, cudnnRNNDescriptor_t, cudnnRNNInputMode_t, cudnnRNNMode_t,
    cudnnSetRNNDescriptor_v8, CUDNN_RNN_PADDED_IO_ENABLED,
};

use super::error::ZenuCudnnError;

pub enum RNNAlgo {
    Standard,
    PersistStatic,
    PersistDynamic,
}

pub enum RNNCell {
    LSTM,
    GRU,
    RNNRelu,
    RNNTanh,
}

pub enum RNNBias {
    NoBias,
    SingleInpBias,
    SingleRecBias,
    DoubleBias,
}

pub enum MathType {
    Default,
    TensorOp,
    TensorOpAllowConversion,
}

#[allow(clippy::too_many_arguments)]
fn rnn_descriptor<Data: 'static, Math: 'static>(
    algo: RNNAlgo,
    cell: RNNCell,
    bias: RNNBias,
    birectional: bool,
    math_type: MathType,
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
        MathType::Default => cudnnMathType_t::CUDNN_DEFAULT_MATH,
        MathType::TensorOp => cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
        MathType::TensorOpAllowConversion => cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
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
            input_size as i32,
            hidden_size as i32,
            hidden_size as i32,
            num_layers as i32,
            dropout,
            CUDNN_RNN_PADDED_IO_ENABLED,
        )
    };

    if status != zenu_cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(ZenuCudnnError::from(status));
    }

    Ok(rnn_desc)
}

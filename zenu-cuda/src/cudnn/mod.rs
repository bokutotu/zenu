use std::any::TypeId;

use zenu_cudnn_sys::*;

use self::error::ZenuCudnnError;

pub mod batch_norm;
pub mod conv;
pub mod error;
pub mod pooling;

pub(crate) fn zenu_cudnn_data_type<T: 'static>() -> cudnnDataType_t {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        cudnnDataType_t::CUDNN_DATA_FLOAT
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        cudnnDataType_t::CUDNN_DATA_DOUBLE
    } else if TypeId::of::<T>() == TypeId::of::<i32>() {
        cudnnDataType_t::CUDNN_DATA_INT32
    } else if TypeId::of::<T>() == TypeId::of::<i64>() {
        cudnnDataType_t::CUDNN_DATA_INT64
    } else {
        panic!("Unsupported data type");
    }
}

pub enum TensorFormat {
    NCHW,
    NHWC,
    NchwVectC,
}

impl From<TensorFormat> for cudnnTensorFormat_t {
    fn from(format: TensorFormat) -> Self {
        match format {
            TensorFormat::NCHW => cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            TensorFormat::NHWC => cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
            TensorFormat::NchwVectC => cudnnTensorFormat_t::CUDNN_TENSOR_NCHW_VECT_C,
        }
    }
}

pub(crate) fn tensor_descriptor_4d<T: 'static>(
    n: i32,
    c: i32,
    h: i32,
    w: i32,
    format: TensorFormat,
) -> Result<cudnnTensorDescriptor_t, ZenuCudnnError> {
    let data_type = zenu_cudnn_data_type::<T>();
    let format = format.into();
    let mut tensor: cudnnTensorDescriptor_t = std::ptr::null_mut();
    unsafe {
        let status = cudnnCreateTensorDescriptor(&mut tensor as *mut cudnnTensorDescriptor_t);
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        let status = cudnnSetTensor4dDescriptor(tensor, format, data_type, n, c, h, w);
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(tensor)
}

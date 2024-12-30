use std::any::TypeId;

use zenu_cudnn_sys::{
    cudnnCreate, cudnnCreateFilterDescriptor, cudnnCreateTensorDescriptor, cudnnDataType_t,
    cudnnFilterDescriptor_t, cudnnHandle_t, cudnnSetFilter4dDescriptor, cudnnSetTensor4dDescriptor,
    cudnnSetTensorNdDescriptor, cudnnStatus_t, cudnnTensorDescriptor_t, cudnnTensorFormat_t,
};

use self::error::ZenuCudnnError;

pub mod batch_norm;
pub mod dropout;
pub mod error;
pub mod graph_batchnorm;
pub mod graph_conv;
pub mod pooling;
pub mod rnn;

mod graph_utils;

pub fn zenu_create_cudnn_handle() -> Result<cudnnHandle_t, ZenuCudnnError> {
    let mut handle: cudnnHandle_t = std::ptr::null_mut();
    unsafe {
        let status = cudnnCreate(std::ptr::from_mut(&mut handle));
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(handle)
}

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
        let status = cudnnCreateTensorDescriptor(std::ptr::from_mut(&mut tensor));
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

pub(crate) fn tensor_descriptor_2d<T: 'static>(
    witdh: i32,
    height: i32,
) -> Result<cudnnTensorDescriptor_t, ZenuCudnnError> {
    let data_type = zenu_cudnn_data_type::<T>();
    let mut tensor: cudnnTensorDescriptor_t = std::ptr::null_mut();
    unsafe {
        let status = cudnnCreateTensorDescriptor(std::ptr::from_mut(&mut tensor));
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        let dim = [witdh, height];
        let stride = [height, 1];
        let status =
            cudnnSetTensorNdDescriptor(tensor, data_type, 2, dim.as_ptr(), stride.as_ptr());
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(tensor)
}

pub(crate) fn tensor_descriptor_nd<T: 'static>(
    dims: &[i32],
    strides: &[i32],
) -> Result<cudnnTensorDescriptor_t, ZenuCudnnError> {
    let data_type = zenu_cudnn_data_type::<T>();
    let mut tensor: cudnnTensorDescriptor_t = std::ptr::null_mut();
    unsafe {
        let status = cudnnCreateTensorDescriptor(std::ptr::from_mut(&mut tensor));
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        let status = cudnnSetTensorNdDescriptor(
            tensor,
            data_type,
            i32::try_from(dims.len()).unwrap(),
            dims.as_ptr(),
            strides.as_ptr(),
        );
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(tensor)
}

pub(crate) fn filter_descriptor_4d<T: 'static>(
    k: i32,
    c: i32,
    h: i32,
    w: i32,
    format: TensorFormat,
) -> Result<cudnnFilterDescriptor_t, ZenuCudnnError> {
    let data_type = zenu_cudnn_data_type::<T>();
    let format = format.into();
    let mut filter: cudnnFilterDescriptor_t = std::ptr::null_mut();
    unsafe {
        // let status = cudnnCreateFilterDescriptor(&mut filter as *mut cudnnFilterDescriptor_t);
        let status = cudnnCreateFilterDescriptor(std::ptr::from_mut(&mut filter));
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        let status = cudnnSetFilter4dDescriptor(filter, data_type, format, k, c, h, w);
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(filter)
}

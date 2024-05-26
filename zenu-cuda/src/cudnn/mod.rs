use std::{any::TypeId, ptr::NonNull};

use zenu_cudnn_sys::{
    cudnnCreateFilterDescriptor, cudnnCreateTensorDescriptor, cudnnDataType_t,
    cudnnFilterDescriptor_t, cudnnSetFilter4dDescriptor, cudnnSetTensor4dDescriptor,
    cudnnTensorDescriptor_t, cudnnTensorFormat_t,
};

pub mod error;

pub fn zenu_cudnn_data_type<T: 'static>() -> cudnnDataType_t {
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

pub fn zenu_cudnn_tensor_discriptor<T: 'static>(
    n: i32,
    c: i32,
    h: i32,
    w: i32,
) -> cudnnTensorDescriptor_t {
    let data_type = zenu_cudnn_data_type::<T>();
    let mut tensor_desc: cudnnTensorDescriptor_t = std::ptr::null_mut();
    unsafe {
        cudnnCreateTensorDescriptor(&mut tensor_desc);
        cudnnSetTensor4dDescriptor(
            tensor_desc,
            zenu_cudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            data_type,
            n,
            c,
            h,
            w,
        );
    }
    tensor_desc
}

pub fn zenu_cudnn_fileter_discriptor<T: 'static>(
    k: i32,
    c: i32,
    h: i32,
    w: i32,
    layout: cudnnTensorFormat_t,
) -> cudnnFilterDescriptor_t {
    let data_type = zenu_cudnn_data_type::<T>();
    let mut tensor_desc: cudnnFilterDescriptor_t = std::ptr::null_mut();
    unsafe {
        cudnnCreateFilterDescriptor(&mut tensor_desc);
        cudnnSetFilter4dDescriptor(tensor_desc, data_type, layout, k, c, h, w);
    }
    tensor_desc
}

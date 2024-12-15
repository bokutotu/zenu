use zenu_cudnn_frontend_wrapper_sys::{
    CudnnFrontendDataType_t, CudnnFrontendError_t, CudnnTensorShapeStride,
};

#[expect(clippy::cast_possible_wrap)]
pub fn shape_stride_to_cudnn(shape: &[usize], stride: &[usize]) -> CudnnTensorShapeStride {
    let num_dims = shape.len();
    let mut dims = [0_i64; 8];
    let mut strides = [0_i64; 8];
    for i in 0..num_dims {
        dims[i] = shape[i] as i64;
        strides[i] = stride[i] as i64;
    }
    CudnnTensorShapeStride {
        num_dims,
        dims,
        strides,
    }
}

pub fn get_cudnn_frontend_type<T>() -> CudnnFrontendDataType_t {
    if std::any::type_name::<T>() == "f32" {
        CudnnFrontendDataType_t::DATA_TYPE_FLOAT
    } else if std::any::type_name::<T>() == "f64" {
        CudnnFrontendDataType_t::DATA_TYPE_DOUBLE
    } else {
        panic!("Unsupported data type");
    }
}

pub fn success_or_panic(status: CudnnFrontendError_t) {
    assert!(
        status == CudnnFrontendError_t::SUCCESS,
        "Cudnn frontend error: {status:?}"
    );
}

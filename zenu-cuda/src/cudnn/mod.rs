use std::any::TypeId;

use zenu_cudnn_sys::*;

use crate::ZENU_CUDA_STATE;

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

fn tensor_descriptor<T: 'static>(
    n: i32,
    c: i32,
    h: i32,
    w: i32,
    format: TensorFormat,
) -> cudnnTensorDescriptor_t {
    let data_type = zenu_cudnn_data_type::<T>();
    let format = format.into();
    let mut tensor = std::ptr::null_mut();
    unsafe {
        cudnnCreateTensorDescriptor(&mut tensor as *mut cudnnTensorDescriptor_t);
        cudnnSetTensor4dDescriptor(tensor, format, data_type, n, c, h, w);
    }
    tensor
}

fn filter_descriptor<T: 'static>(
    k: i32,
    c: i32,
    h: i32,
    w: i32,
    format: TensorFormat,
) -> cudnnFilterDescriptor_t {
    let data_type = zenu_cudnn_data_type::<T>();
    let format = format.into();
    let mut filter = std::ptr::null_mut();
    unsafe {
        cudnnCreateFilterDescriptor(&mut filter as *mut cudnnFilterDescriptor_t);
        cudnnSetFilter4dDescriptor(filter, data_type, format, k, c, h, w);
    }
    filter
}

fn convolution_descriptor(
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dilation_h: i32,
    dilation_w: i32,
) -> cudnnConvolutionDescriptor_t {
    let mut conv = std::ptr::null_mut();
    unsafe {
        cudnnCreateConvolutionDescriptor(&mut conv as *mut cudnnConvolutionDescriptor_t);
        cudnnSetConvolution2dDescriptor(
            conv,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            zenu_cudnn_data_type::<f32>(),
        );
    }
    conv
}

fn convolution_algorithm(
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    requested_algo_count: usize,
) -> cudnnConvolutionFwdAlgo_t {
    let state = ZENU_CUDA_STATE.lock().unwrap();
    let handle = state.get_cudnn();
    let mut returned_algo_count = 0;
    let algorithm = unsafe {
        let mut algorithm: Vec<cudnnConvolutionFwdAlgoPerf_t> = Vec::new();
        // allow use tensor core
        cudnnSetConvolutionMathType(conv, cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

        cudnnGetConvolutionForwardAlgorithm_v7(
            handle.as_ptr(),
            input,
            filter,
            conv,
            output,
            requested_algo_count as i32,
            &mut returned_algo_count as *mut i32,
            algorithm.as_mut_ptr() as *mut cudnnConvolutionFwdAlgoPerf_t,
        );
        algorithm[0].algo
    };
    algorithm
}

fn convolution_workspace(
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    algorithm: cudnnConvolutionFwdAlgo_t,
) -> ConvWorkspace {
    let state = ZENU_CUDA_STATE.lock().unwrap();
    let handle = state.get_cudnn();
    let mut workspace_size = 0;
    unsafe {
        cudnnGetConvolutionForwardWorkspaceSize(
            handle.as_ptr(),
            input,
            filter,
            conv,
            output,
            algorithm,
            &mut workspace_size as *mut usize,
        );
        let mut workspace = std::ptr::null_mut();
        cudaMalloc(&mut workspace as *mut *mut libc::c_void, workspace_size);
        ConvWorkspace {
            workspace,
            workspace_size,
        }
    }
}

#[derive(Debug)]
pub struct ConvWorkspace {
    workspace: *mut libc::c_void,
    workspace_size: usize,
}

#[derive(Debug)]
pub struct ConvDescriptor {
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    algorithm: cudnnConvolutionFwdAlgo_t,
    workspace: ConvWorkspace,
}

impl ConvDescriptor {
    pub fn forward<T: 'static>(
        &self,
        alpha: T,
        input: *const T,
        filter: *const T,
        beta: T,
        output: *mut T,
    ) {
        let state = ZENU_CUDA_STATE.lock().unwrap();
        let handle = state.get_cudnn();
        unsafe {
            cudnnConvolutionForward(
                handle.as_ptr(),
                &alpha as *const T as *const libc::c_void,
                self.input,
                input as *const libc::c_void,
                self.filter,
                filter as *const libc::c_void,
                self.conv,
                self.algorithm,
                self.workspace.workspace,
                self.workspace.workspace_size,
                &beta as *const T as *const libc::c_void,
                self.output,
                output as *mut libc::c_void,
            );
        }
    }
}

impl Drop for ConvDescriptor {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyTensorDescriptor(self.input);
            cudnnDestroyFilterDescriptor(self.filter);
            cudnnDestroyConvolutionDescriptor(self.conv);
            cudnnDestroyTensorDescriptor(self.output);
            cudaFree(self.workspace.workspace);
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq, Hash)]
pub struct ConvolutionBuilder {
    input: Option<cudnnTensorDescriptor_t>,
    filter: Option<cudnnFilterDescriptor_t>,
    conv: Option<cudnnConvolutionDescriptor_t>,
    output: Option<cudnnTensorDescriptor_t>,
    algorithm: Option<cudnnConvolutionFwdAlgo_t>,
}

impl ConvolutionBuilder {
    pub fn input<T: 'static>(self, n: i32, c: i32, h: i32, w: i32, format: TensorFormat) -> Self {
        let input = tensor_descriptor::<T>(n, c, h, w, format);
        Self {
            input: Some(input),
            ..self
        }
    }

    pub fn filter<T: 'static>(self, k: i32, c: i32, h: i32, w: i32, format: TensorFormat) -> Self {
        let filter = filter_descriptor::<T>(k, c, h, w, format);
        Self {
            filter: Some(filter),
            ..self
        }
    }

    pub fn conv(
        self,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
    ) -> Self {
        let conv = convolution_descriptor(pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
        Self {
            conv: Some(conv),
            ..self
        }
    }

    pub fn output<T: 'static>(self, n: i32, c: i32, h: i32, w: i32, format: TensorFormat) -> Self {
        let output = tensor_descriptor::<T>(n, c, h, w, format);
        Self {
            output: Some(output),
            ..self
        }
    }

    pub fn algorithm(self, requested_algo_count: usize) -> Self {
        let input = self.input.unwrap();
        let filter = self.filter.unwrap();
        let conv = self.conv.unwrap();
        let output = self.output.unwrap();
        let algorithm = convolution_algorithm(input, filter, conv, output, requested_algo_count);
        Self {
            algorithm: Some(algorithm),
            ..self
        }
    }

    pub fn build(self) -> ConvDescriptor {
        let input = self.input.unwrap();
        let filter = self.filter.unwrap();
        let conv = self.conv.unwrap();
        let output = self.output.unwrap();
        let algorithm = self.algorithm.unwrap();
        let workspace = convolution_workspace(input, filter, conv, output, algorithm);
        ConvDescriptor {
            input,
            filter,
            conv,
            output,
            algorithm,
            workspace,
        }
    }
}

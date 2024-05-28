use std::any::TypeId;

use zenu_cudnn_sys::*;

use crate::ZENU_CUDA_STATE;

use self::error::ZenuCudnnError;

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

fn filter_descriptor<T: 'static>(
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
        let status = cudnnCreateFilterDescriptor(&mut filter as *mut cudnnFilterDescriptor_t);
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

fn convolution_descriptor(
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dilation_h: i32,
    dilation_w: i32,
) -> Result<cudnnConvolutionDescriptor_t, ZenuCudnnError> {
    let mut conv: cudnnConvolutionDescriptor_t = std::ptr::null_mut();
    unsafe {
        let status =
            cudnnCreateConvolutionDescriptor(&mut conv as *mut cudnnConvolutionDescriptor_t);
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        let status = cudnnSetConvolution2dDescriptor(
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
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(conv)
}

fn convolution_algorithm(
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    requested_algo_count: usize,
) -> Result<cudnnConvolutionFwdAlgo_t, ZenuCudnnError> {
    let state = ZENU_CUDA_STATE.lock().unwrap();
    let handle = state.get_cudnn();
    let mut returned_algo_count = 0;
    unsafe {
        let mut algorithm: Vec<cudnnConvolutionFwdAlgoPerf_t> =
            Vec::with_capacity(requested_algo_count);
        for _ in 0..requested_algo_count {
            algorithm.push(cudnnConvolutionFwdAlgoPerf_t::default());
        }

        // enable tensor core
        cudnnSetConvolutionMathType(conv, cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

        let state = cudnnGetConvolutionForwardAlgorithm_v7(
            handle.as_ptr(),
            input,
            filter,
            conv,
            output,
            requested_algo_count as i32,
            &mut returned_algo_count as *mut i32,
            algorithm.as_mut_ptr() as *mut cudnnConvolutionFwdAlgoPerf_t,
        );
        if state != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(state));
        }

        Ok(algorithm[0].algo)
    }
}

fn convolution_workspace(
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    algorithm: cudnnConvolutionFwdAlgo_t,
) -> Result<ConvWorkspace, ZenuCudnnError> {
    let state = ZENU_CUDA_STATE.lock().unwrap();
    let handle = state.get_cudnn();
    let mut workspace_size = 0;
    unsafe {
        let status = cudnnGetConvolutionForwardWorkspaceSize(
            handle.as_ptr(),
            input,
            filter,
            conv,
            output,
            algorithm,
            &mut workspace_size as *mut usize,
        );
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            panic!("Failed to get convolution forward workspace size");
        }
        let mut workspace = std::ptr::null_mut();
        let status = cudaMalloc(&mut workspace as *mut *mut libc::c_void, workspace_size);
        if status != cudaError_t::cudaSuccess {
            panic!("Failed to allocate convolution forward workspace");
        }
        Ok(ConvWorkspace {
            workspace,
            workspace_size,
        })
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
    pub fn input<T: 'static>(
        self,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        format: TensorFormat,
    ) -> Result<Self, ZenuCudnnError> {
        let input = tensor_descriptor::<T>(n, c, h, w, format)?;
        Ok(Self {
            input: Some(input),
            ..self
        })
    }

    pub fn filter<T: 'static>(
        self,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
        format: TensorFormat,
    ) -> Result<Self, ZenuCudnnError> {
        let filter = filter_descriptor::<T>(k, c, h, w, format)?;
        Ok(Self {
            filter: Some(filter),
            ..self
        })
    }

    pub fn conv(
        self,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
    ) -> Result<Self, ZenuCudnnError> {
        let conv =
            convolution_descriptor(pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w)?;
        Ok(Self {
            conv: Some(conv),
            ..self
        })
    }

    pub fn output<T: 'static>(
        self,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        format: TensorFormat,
    ) -> Result<Self, ZenuCudnnError> {
        let output = tensor_descriptor::<T>(n, c, h, w, format)?;
        Ok(Self {
            output: Some(output),
            ..self
        })
    }

    pub fn algorithm(self, requested_algo_count: usize) -> Result<Self, ZenuCudnnError> {
        let input = self.input.unwrap();
        let filter = self.filter.unwrap();
        let conv = self.conv.unwrap();
        let output = self.output.unwrap();
        let algorithm = convolution_algorithm(input, filter, conv, output, requested_algo_count)?;
        Ok(Self {
            algorithm: Some(algorithm),
            ..self
        })
    }

    pub fn build(self) -> Result<ConvDescriptor, ZenuCudnnError> {
        let input = self.input.unwrap();
        let filter = self.filter.unwrap();
        let conv = self.conv.unwrap();
        let output = self.output.unwrap();
        let algorithm = self.algorithm.unwrap();
        let workspace = convolution_workspace(input, filter, conv, output, algorithm)?;
        Ok(ConvDescriptor {
            input,
            filter,
            conv,
            output,
            algorithm,
            workspace,
        })
    }
}

#[cfg(test)]
mod cudnn {
    use super::*;

    // #[test]
    // fn test_convolution() {
    //     // int n = 1, c = 3, h = 32, w = 32;
    //     // int k = 8, kh = 5, kw = 5;
    //     // int pad_h = 1, pad_w = 1, stride_h = 1, stride_w = 1;
    //     let n = 1;
    //     let c = 3;
    //     let h = 32;
    //     let w = 32;
    //     let k = 8;
    //     let kh = 5;
    //     let kw = 5;
    //     let pad_h = 1;
    //     let pad_w = 1;
    //     let stride_h = 1;
    //     let stride_w = 1;
    //
    //     let conv = ConvolutionBuilder::default()
    //         .input::<f32>(n, c, h, w, TensorFormat::NCHW)
    //         .unwrap()
    //         .filter::<f32>(k, c, kh, kw, TensorFormat::NCHW)
    //         .unwrap()
    //         .conv(pad_h, pad_w, stride_h, stride_w, 1, 1)
    //         .unwrap()
    //         .output::<f32>(n, k, h, w, TensorFormat::NCHW)
    //         .unwrap()
    //         .algorithm(1)
    //         .unwrap()
    //         .build()
    //         .unwrap();
    // }
    #[test]
    fn test_convolution() {
        let n = 1;
        let c = 3;
        let h = 32;
        let w = 32;
        let k = 8;
        let kh = 5;
        let kw = 5;
        let pad_h = 1;
        let pad_w = 1;
        let stride_h = 1;
        let stride_w = 1;

        // 畳み込み後の出力テンソルのサイズ
        let out_h = (h + 2 * pad_h - kh) / stride_h + 1;
        let out_w = (w + 2 * pad_w - kw) / stride_w + 1;

        let conv = ConvolutionBuilder::default()
            .input::<f32>(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .filter::<f32>(k, c, kh, kw, TensorFormat::NCHW)
            .unwrap()
            .conv(pad_h, pad_w, stride_h, stride_w, 1, 1)
            .unwrap()
            .output::<f32>(n, k, out_h, out_w, TensorFormat::NCHW) // ここで出力テンソルのサイズを変更
            .unwrap()
            .algorithm(1)
            .unwrap()
            .build()
            .unwrap();
    }
}

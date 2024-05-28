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
    use crate::runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind};

    use super::*;

    #[test]
    fn test_convolution() {
        let n = 1;
        let c = 3;
        let h = 5;
        let w = 5;
        let k = 3;
        let kh = 3;
        let kw = 3;
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

        // create input tensor
        let mut input_cpu = Vec::new();
        for idx in 0..n * c * h * w {
            input_cpu.push(idx as f32);
        }
        let input_gpu = cuda_malloc::<f32>((n * c * h * w) as usize).unwrap();
        cuda_copy(
            input_gpu,
            input_cpu.as_ptr(),
            (n * c * h * w) as usize,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        // create filter tensor
        let mut filter_cpu = Vec::new();
        for idx in 0..k * c * kh * kw {
            filter_cpu.push(idx as f32);
        }
        let filter_gpu = cuda_malloc::<f32>((k * c * kh * kw) as usize).unwrap();
        cuda_copy(
            filter_gpu,
            filter_cpu.as_ptr(),
            (k * c * kh * kw) as usize,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        // create output tensor
        let output_gpu = cuda_malloc::<f32>((n * k * out_h * out_w) as usize).unwrap();

        // execute convolution
        conv.forward(1.0, input_gpu, filter_gpu, 0.0, output_gpu);

        // copy output tensor to cpu
        let mut output_cpu = Vec::new();
        for _ in 0..n * k * out_h * out_w {
            output_cpu.push(0.0);
        }
        cuda_copy(
            output_cpu.as_mut_ptr(),
            output_gpu,
            (n * k * out_h * out_w) as usize,
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();

        // check output tensor
        let ans = vec![
            6888, 10218, 10479, 10740, 7056, 10296, 15219, 15570, 15921, 10422, 11511, 16974,
            17325, 17676, 11547, 12726, 18729, 19080, 19431, 12672, 8040, 11784, 11991, 12198,
            7920, 15960, 24069, 24816, 25563, 17100, 25119, 37818, 38898, 39978, 26703, 28764,
            43218, 44298, 45378, 30258, 32409, 48618, 49698, 50778, 33813, 21972, 32925, 33618,
            34311, 22824, 25032, 37920, 39153, 40386, 27144, 39942, 60417, 62226, 64035, 42984,
            46017, 69462, 71271, 73080, 48969, 52092, 78507, 80316, 82125, 54954, 35904, 54066,
            55245, 56424, 37728,
        ];
        let ans = ans.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        assert_eq!(output_cpu, ans);
    }
}

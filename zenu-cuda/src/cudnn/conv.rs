use super::{error::ZenuCudnnError, tensor_descriptor_4d, zenu_cudnn_data_type, TensorFormat};

use crate::ZENU_CUDA_STATE;

use std::cell::UnsafeCell;

use zenu_cudnn_sys::*;

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

fn convolution_backward_data_algorithm(
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    requested_algo_count: usize,
) -> Result<cudnnConvolutionBwdDataAlgo_t, ZenuCudnnError> {
    let state = ZENU_CUDA_STATE.lock().unwrap();
    let handle = state.get_cudnn();
    let mut returned_algo_count = 0;
    unsafe {
        let mut algorithm: Vec<cudnnConvolutionBwdDataAlgoPerf_t> = Vec::with_capacity(1);
        for _ in 0..requested_algo_count {
            algorithm.push(cudnnConvolutionBwdDataAlgoPerf_t::default());
        }

        // enable tensor core
        cudnnSetConvolutionMathType(conv, cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

        let state = cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle.as_ptr(),
            filter,
            input,
            conv,
            output,
            requested_algo_count as i32,
            &mut returned_algo_count as *mut i32,
            algorithm.as_mut_ptr() as *mut cudnnConvolutionBwdDataAlgoPerf_t,
        );
        if state != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(state));
        }

        Ok(algorithm[0].algo)
    }
}

fn convolution_backward_filter_algorithm(
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    requested_algo_count: usize,
) -> Result<cudnnConvolutionBwdFilterAlgo_t, ZenuCudnnError> {
    let state = ZENU_CUDA_STATE.lock().unwrap();
    let handle = state.get_cudnn();
    let mut returned_algo_count = 0;
    unsafe {
        let mut algorithm: Vec<cudnnConvolutionBwdFilterAlgoPerf_t> =
            Vec::with_capacity(requested_algo_count);
        for _ in 0..requested_algo_count {
            algorithm.push(cudnnConvolutionBwdFilterAlgoPerf_t::default());
        }

        // enable tensor core
        cudnnSetConvolutionMathType(conv, cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

        let state = cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            handle.as_ptr(),
            input,
            output,
            conv,
            filter,
            requested_algo_count as i32,
            &mut returned_algo_count as *mut i32,
            algorithm.as_mut_ptr() as *mut cudnnConvolutionBwdFilterAlgoPerf_t,
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
) -> Result<Workspace, ZenuCudnnError> {
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
        Ok(Workspace::new(workspace_size))
    }
}

fn convolution_backward_data_workspace(
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    algorithm: cudnnConvolutionBwdDataAlgo_t,
) -> Result<Workspace, ZenuCudnnError> {
    let state = ZENU_CUDA_STATE.lock().unwrap();
    let handle = state.get_cudnn();
    let mut workspace_size = 0;
    unsafe {
        let status = cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle.as_ptr(),
            filter,
            output,
            conv,
            input,
            algorithm,
            &mut workspace_size as *mut usize,
        );
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            panic!("Failed to get convolution backward data workspace size");
        }
        Ok(Workspace::new(workspace_size))
    }
}

fn convolution_backward_filter_workspace(
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    algorithm: cudnnConvolutionBwdFilterAlgo_t,
) -> Result<Workspace, ZenuCudnnError> {
    let state = ZENU_CUDA_STATE.lock().unwrap();
    let handle = state.get_cudnn();
    let mut workspace_size = 0;
    unsafe {
        let status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle.as_ptr(),
            input,
            output,
            conv,
            filter,
            algorithm,
            &mut workspace_size as *mut usize,
        );
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            panic!("Failed to get convolution backward filter workspace size");
        }
        Ok(Workspace::new(workspace_size))
    }
}

#[derive(Debug)]
pub struct Workspace {
    workspace: UnsafeCell<Option<*mut libc::c_void>>,
    workspace_size: usize,
}

impl Workspace {
    pub fn new(workspace_size: usize) -> Self {
        Self {
            workspace: UnsafeCell::new(None),
            workspace_size,
        }
    }

    pub fn workspace(&self) -> *mut libc::c_void {
        let workspace = unsafe { &mut *self.workspace.get() };
        if workspace.is_none() {
            let ptr = unsafe {
                let mut ptr = std::ptr::null_mut();
                cudaMalloc(&mut ptr as *mut *mut libc::c_void, self.workspace_size);
                ptr
            };
            *workspace = Some(ptr);
        }
        workspace.unwrap()
    }

    pub fn free_workspace(&self) {
        let workspace = unsafe { &mut *self.workspace.get() };
        if let Some(ptr) = workspace.take() {
            unsafe {
                cudaFree(ptr);
            }
            *workspace = None;
        }
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        let workspace = unsafe { &mut *self.workspace.get() };
        if let Some(ptr) = workspace.take() {
            unsafe {
                cudaFree(ptr);
            }
        }
    }
}

#[derive(Debug)]
pub struct ConvDescriptor {
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    algorithm: cudnnConvolutionFwdAlgo_t,
    workspace: Workspace,
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
                self.workspace.workspace(),
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
        }
    }
}

pub struct ConvolutionBackwardData {
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    algorithm: cudnnConvolutionBwdDataAlgo_t,
    workspace: Workspace,
}

impl ConvolutionBackwardData {
    pub fn backward_data<T: 'static>(
        &self,
        alpha: T,
        filter: *const T,
        output: *const T,
        beta: T,
        input: *mut T,
    ) {
        let state = ZENU_CUDA_STATE.lock().unwrap();
        let handle = state.get_cudnn();
        unsafe {
            cudnnConvolutionBackwardData(
                handle.as_ptr(),
                &alpha as *const T as *const libc::c_void,
                self.filter,
                filter as *const libc::c_void,
                self.output,
                output as *const libc::c_void,
                self.conv,
                self.algorithm,
                self.workspace.workspace(),
                self.workspace.workspace_size,
                &beta as *const T as *const libc::c_void,
                self.input,
                input as *mut libc::c_void,
            );
        }
    }
}

impl Drop for ConvolutionBackwardData {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyTensorDescriptor(self.input);
            cudnnDestroyFilterDescriptor(self.filter);
            cudnnDestroyConvolutionDescriptor(self.conv);
            cudnnDestroyTensorDescriptor(self.output);
        }
    }
}

pub struct ConvolutionBackwardFilter {
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    algorithm: cudnnConvolutionBwdFilterAlgo_t,
    workspace: Workspace,
}

impl ConvolutionBackwardFilter {
    pub fn backward_filter<T: 'static>(
        &self,
        alpha: T,
        input: *const T,
        d_output: *const T,
        beta: T,
        filter: *mut T,
    ) {
        let state = ZENU_CUDA_STATE.lock().unwrap();
        let handle = state.get_cudnn();
        unsafe {
            cudnnConvolutionBackwardFilter(
                handle.as_ptr(),
                &alpha as *const T as *const libc::c_void,
                self.input,
                input as *const libc::c_void,
                self.output,
                d_output as *const libc::c_void,
                self.conv,
                self.algorithm,
                self.workspace.workspace(),
                self.workspace.workspace_size,
                &beta as *const T as *const libc::c_void,
                self.filter,
                filter as *mut libc::c_void,
            );
        }
    }
}

impl Drop for ConvolutionBackwardFilter {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyTensorDescriptor(self.input);
            cudnnDestroyFilterDescriptor(self.filter);
            cudnnDestroyConvolutionDescriptor(self.conv);
            cudnnDestroyTensorDescriptor(self.output);
        }
    }
}

macro_rules! impl_convolution {
    ($desc_name:ident, $algo:ident, $builder:ident, $algo_func:ident, $workspace:ident) => {
        #[derive(Default)]
        pub struct $builder {
            input: Option<cudnnTensorDescriptor_t>,
            filter: Option<cudnnFilterDescriptor_t>,
            conv: Option<cudnnConvolutionDescriptor_t>,
            output: Option<cudnnTensorDescriptor_t>,
            algorithm: Option<$algo>,
        }

        impl $builder {
            pub fn input<T: 'static>(
                self,
                n: i32,
                c: i32,
                h: i32,
                w: i32,
                format: TensorFormat,
            ) -> Result<Self, ZenuCudnnError> {
                let input = tensor_descriptor_4d::<T>(n, c, h, w, format)?;
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
                let conv = convolution_descriptor(
                    pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                )?;
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
                let output = tensor_descriptor_4d::<T>(n, c, h, w, format)?;
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
                let algorithm = $algo_func(input, filter, conv, output, requested_algo_count)?;
                Ok(Self {
                    algorithm: Some(algorithm),
                    ..self
                })
            }

            pub fn build(self) -> Result<$desc_name, ZenuCudnnError> {
                let input = self.input.unwrap();
                let filter = self.filter.unwrap();
                let conv = self.conv.unwrap();
                let output = self.output.unwrap();
                let algorithm = self.algorithm.unwrap();
                let workspace = $workspace(input, filter, conv, output, algorithm)?;
                Ok($desc_name {
                    input,
                    filter,
                    conv,
                    output,
                    algorithm,
                    workspace,
                })
            }
        }
    };
}
impl_convolution!(
    ConvDescriptor,
    cudnnConvolutionFwdAlgo_t,
    ConvolutionBuilder,
    convolution_algorithm,
    convolution_workspace
);
impl_convolution!(
    ConvolutionBackwardData,
    cudnnConvolutionBwdDataAlgo_t,
    ConvolutionBackwardDataBuilder,
    convolution_backward_data_algorithm,
    convolution_backward_data_workspace
);
impl_convolution!(
    ConvolutionBackwardFilter,
    cudnnConvolutionBwdFilterAlgo_t,
    ConvolutionBackwardFilterBuilder,
    convolution_backward_filter_algorithm,
    convolution_backward_filter_workspace
);

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

    #[test]
    fn bkwd_data() {
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

        let conv = ConvolutionBackwardDataBuilder::default()
            .input::<f32>(n, c, out_h, out_w, TensorFormat::NCHW)
            .unwrap()
            .filter::<f32>(k, c, kh, kw, TensorFormat::NCHW)
            .unwrap()
            .conv(pad_h, pad_w, stride_h, stride_w, 1, 1)
            .unwrap()
            .output::<f32>(n, k, h, w, TensorFormat::NCHW) // ここで出力テンソルのサイズを変更
            .unwrap()
            .algorithm(5)
            .unwrap()
            .build()
            .unwrap();

        let mut input_cpu = Vec::new();
        for idx in 0..n * c * out_h * out_w {
            input_cpu.push(idx as f32);
        }

        let mut filter_cpu = Vec::new();
        for idx in 0..k * c * kh * kw {
            filter_cpu.push(idx as f32);
        }

        let input_gpu = cuda_malloc::<f32>((n * c * out_h * out_w) as usize).unwrap();
        let filter_gpu = cuda_malloc::<f32>((k * c * kh * kw) as usize).unwrap();
        let output_gpu = cuda_malloc::<f32>((n * k * h * w) as usize).unwrap();

        cuda_copy(
            input_gpu,
            input_cpu.as_ptr(),
            (n * c * out_h * out_w) as usize,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        cuda_copy(
            filter_gpu,
            filter_cpu.as_ptr(),
            (k * c * kh * kw) as usize,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        conv.backward_data(1.0, filter_gpu, input_gpu, 0.0, output_gpu);

        let mut output_cpu = Vec::new();
        for _ in 0..n * k * h * w {
            output_cpu.push(0.0);
        }
        cuda_copy(
            output_cpu.as_mut_ptr(),
            output_gpu,
            (n * k * h * w) as usize,
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();
        let ans = vec![
            15096.0, 23154.0, 23685.0, 24216.0, 16512.0, 24660.0, 37809.0, 38646.0, 39483.0,
            26910.0, 27405.0, 41994.0, 42831.0, 43668.0, 29745.0, 30150.0, 46179.0, 47016.0,
            47853.0, 32580.0, 21864.0, 33468.0, 34053.0, 34638.0, 23568.0, 18120.0, 27771.0,
            28464.0, 29157.0, 19860.0, 29601.0, 45342.0, 46422.0, 47502.0, 32337.0, 33156.0,
            50742.0, 51822.0, 52902.0, 35982.0, 36711.0, 56142.0, 57222.0, 58302.0, 39627.0,
            26508.0, 40515.0, 41262.0, 42009.0, 28536.0, 21144.0, 32388.0, 33243.0, 34098.0,
            23208.0, 34542.0, 52875.0, 54198.0, 55521.0, 37764.0, 38907.0, 59490.0, 60813.0,
            62136.0, 42219.0, 43272.0, 66105.0, 67428.0, 68751.0, 46674.0, 31152.0, 47562.0,
            48471.0, 49380.0, 33504.0,
        ];
        assert_eq!(output_cpu, ans);
    }

    #[test]
    fn bkwd_filter() {
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

        let conv = ConvolutionBackwardFilterBuilder::default()
            .input::<f32>(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .filter::<f32>(k, c, kh, kw, TensorFormat::NCHW)
            .unwrap()
            .conv(pad_h, pad_w, stride_h, stride_w, 1, 1)
            .unwrap()
            .output::<f32>(n, k, out_h, out_w, TensorFormat::NCHW)
            .unwrap()
            .algorithm(1)
            .unwrap()
            .build()
            .unwrap();

        let mut input_cpu = Vec::new();
        for idx in 0..n * c * h * w {
            input_cpu.push(idx as f32);
        }

        let mut d_output_cpu = Vec::new();
        for idx in 0..n * k * out_h * out_w {
            d_output_cpu.push((idx % 10) as f32);
        }

        let input_gpu = cuda_malloc::<f32>((n * c * h * w) as usize).unwrap();
        let filter_gpu = cuda_malloc::<f32>((k * c * kh * kw) as usize).unwrap();
        let output_gpu = cuda_malloc::<f32>((n * k * out_h * out_w) as usize).unwrap();

        cuda_copy(
            input_gpu,
            input_cpu.as_ptr(),
            (n * c * h * w) as usize,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        cuda_copy(
            output_gpu,
            d_output_cpu.as_ptr(),
            (n * k * out_h * out_w) as usize,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        conv.backward_filter(1.0, input_gpu, output_gpu, 0.0, filter_gpu);

        let mut filter_cpu = Vec::new();
        for _ in 0..k * c * kh * kw {
            filter_cpu.push(0.0);
        }
        cuda_copy(
            filter_cpu.as_mut_ptr(),
            filter_gpu,
            (k * c * kh * kw) as usize,
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();

        let ans = vec![
            640.0, 770.0, 560.0, 1060.0, 1250.0, 900.0, 1240.0, 1470.0, 1080.0, 2640.0, 3020.0,
            2160.0, 3310.0, 3750.0, 2650.0, 3240.0, 3720.0, 2680.0, 4640.0, 5270.0, 3760.0, 5560.0,
            6250.0, 4400.0, 5240.0, 5970.0, 4280.0, 840.0, 1020.0, 760.0, 1290.0, 1550.0, 1150.0,
            1040.0, 1220.0, 880.0, 2840.0, 3270.0, 2360.0, 4040.0, 4675.0, 3400.0, 3040.0, 3470.0,
            2480.0, 4840.0, 5520.0, 3960.0, 6790.0, 7800.0, 5650.0, 5040.0, 5720.0, 4080.0, 640.0,
            770.0, 560.0, 1060.0, 1250.0, 900.0, 1240.0, 1470.0, 1080.0, 2640.0, 3020.0, 2160.0,
            3310.0, 3750.0, 2650.0, 3240.0, 3720.0, 2680.0, 4640.0, 5270.0, 3760.0, 5560.0, 6250.0,
            4400.0, 5240.0, 5970.0, 4280.0,
        ];

        assert_eq!(filter_cpu, ans);
    }
}

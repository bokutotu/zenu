use zenu_cuda_runtime_sys::{cudaFree, cudaMalloc};
use zenu_cudnn_sys::{
    cudnnConvolutionBackwardBias, cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter,
    cudnnConvolutionBwdDataAlgoPerf_t, cudnnConvolutionBwdDataAlgo_t,
    cudnnConvolutionBwdFilterAlgoPerf_t, cudnnConvolutionBwdFilterAlgo_t,
    cudnnConvolutionDescriptor_t, cudnnConvolutionForward, cudnnConvolutionFwdAlgoPerf_t,
    cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t, cudnnCreateConvolutionDescriptor,
    cudnnDestroyConvolutionDescriptor, cudnnDestroyFilterDescriptor, cudnnDestroyTensorDescriptor,
    cudnnFilterDescriptor_t, cudnnGetConvolutionBackwardDataAlgorithm_v7,
    cudnnGetConvolutionBackwardDataWorkspaceSize, cudnnGetConvolutionBackwardFilterAlgorithm_v7,
    cudnnGetConvolutionBackwardFilterWorkspaceSize, cudnnGetConvolutionForwardAlgorithm_v7,
    cudnnGetConvolutionForwardWorkspaceSize, cudnnMathType_t, cudnnSetConvolution2dDescriptor,
    cudnnSetConvolutionMathType, cudnnStatus_t, cudnnTensorDescriptor_t,
};

use super::{
    error::ZenuCudnnError, filter_descriptor_4d, tensor_descriptor_4d, zenu_cudnn_data_type,
    TensorFormat,
};

use crate::ZENU_CUDA_STATE;

use std::cell::UnsafeCell;

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
        let status = cudnnCreateConvolutionDescriptor(std::ptr::from_mut(&mut conv));
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
            i32::try_from(requested_algo_count).unwrap(),
            std::ptr::from_mut(&mut returned_algo_count),
            algorithm.as_mut_ptr().cast(),
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
            output,
            conv,
            input,
            i32::try_from(requested_algo_count).unwrap(),
            std::ptr::from_mut(&mut returned_algo_count),
            algorithm.as_mut_ptr().cast(),
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
            i32::try_from(requested_algo_count).unwrap(),
            std::ptr::from_mut(&mut returned_algo_count),
            algorithm.as_mut_ptr().cast(),
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
            std::ptr::from_mut(&mut workspace_size),
        );
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
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
            std::ptr::from_mut(&mut workspace_size),
        );
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
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
) -> Workspace {
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
            std::ptr::from_mut(&mut workspace_size),
        );
        assert!(
            status == cudnnStatus_t::CUDNN_STATUS_SUCCESS,
            "Failed to get convolution backward filter workspace size"
        );
        Workspace::new(workspace_size)
    }
}

#[derive(Debug)]
pub struct Workspace {
    workspace: UnsafeCell<Option<*mut libc::c_void>>,
    workspace_size: usize,
}

impl Workspace {
    #[must_use]
    pub fn new(workspace_size: usize) -> Self {
        Self {
            workspace: UnsafeCell::new(None),
            workspace_size,
        }
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn workspace(&self) -> *mut libc::c_void {
        let workspace = unsafe { &mut *self.workspace.get() };
        if workspace.is_none() {
            let ptr = unsafe {
                let mut ptr = std::ptr::null_mut();
                cudaMalloc(
                    std::ptr::from_mut::<*mut libc::c_void>(&mut ptr),
                    self.workspace_size,
                );
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

pub struct ConvolutionConfig<T> {
    input: cudnnTensorDescriptor_t,
    filter: cudnnFilterDescriptor_t,
    conv: cudnnConvolutionDescriptor_t,
    output: cudnnTensorDescriptor_t,
    fwd_algo: cudnnConvolutionFwdAlgo_t,
    bwd_data_algo: cudnnConvolutionBwdDataAlgo_t,
    bwd_filter_algo: cudnnConvolutionBwdFilterAlgo_t,
    fwd_workspace: Workspace,
    bwd_data_workspace: Workspace,
    bwd_filter_workspace: Workspace,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Copy> ConvolutionConfig<T> {
    #[allow(clippy::missing_panics_doc)]
    pub fn forward(&self, alpha: T, input: *const T, filter: *const T, beta: T, output: *mut T) {
        let state = ZENU_CUDA_STATE.lock().unwrap();
        let handle = state.get_cudnn();
        unsafe {
            cudnnConvolutionForward(
                handle.as_ptr(),
                std::ptr::from_ref(&alpha).cast(),
                self.input,
                input.cast(),
                self.filter,
                filter.cast(),
                self.conv,
                self.fwd_algo,
                self.fwd_workspace.workspace(),
                self.fwd_workspace.workspace_size,
                std::ptr::from_ref(&beta).cast(),
                self.output,
                output.cast(),
            );
        }
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn backward_data(
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
                std::ptr::from_ref(&alpha).cast(),
                self.filter,
                filter.cast(),
                self.output,
                output.cast(),
                self.conv,
                self.bwd_data_algo,
                self.bwd_data_workspace.workspace(),
                self.bwd_data_workspace.workspace_size,
                std::ptr::from_ref(&beta).cast(),
                self.input,
                input.cast(),
            );
        }
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn backward_filter(
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
                std::ptr::from_ref(&alpha).cast(),
                self.input,
                input.cast(),
                self.output,
                d_output.cast(),
                self.conv,
                self.bwd_filter_algo,
                self.bwd_filter_workspace.workspace(),
                self.bwd_filter_workspace.workspace_size,
                std::ptr::from_ref(&beta).cast(),
                self.filter,
                filter.cast(),
            );
        }
    }
}

impl<T> Drop for ConvolutionConfig<T> {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyTensorDescriptor(self.input);
            cudnnDestroyFilterDescriptor(self.filter);
            cudnnDestroyConvolutionDescriptor(self.conv);
            cudnnDestroyTensorDescriptor(self.output);
        }
    }
}

#[derive(Default)]
pub struct ConvolutionBuilder<T> {
    input: Option<cudnnTensorDescriptor_t>,
    filter: Option<cudnnFilterDescriptor_t>,
    conv: Option<cudnnConvolutionDescriptor_t>,
    output: Option<cudnnTensorDescriptor_t>,
    fwd_algo: Option<cudnnConvolutionFwdAlgo_t>,
    bwd_data_algo: Option<cudnnConvolutionBwdDataAlgo_t>,
    bwd_filter_algo: Option<cudnnConvolutionBwdFilterAlgo_t>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: 'static> ConvolutionBuilder<T> {
    #[allow(clippy::missing_errors_doc)]
    pub fn input(
        mut self,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        format: TensorFormat,
    ) -> Result<Self, ZenuCudnnError> {
        self.input = Some(tensor_descriptor_4d::<T>(n, c, h, w, format)?);
        Ok(self)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn filter(
        mut self,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
        format: TensorFormat,
    ) -> Result<Self, ZenuCudnnError> {
        self.filter = Some(filter_descriptor_4d::<T>(k, c, h, w, format)?);
        Ok(self)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn conv(
        mut self,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
    ) -> Result<Self, ZenuCudnnError> {
        self.conv = Some(convolution_descriptor(
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        )?);
        Ok(self)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn output(
        mut self,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        format: TensorFormat,
    ) -> Result<Self, ZenuCudnnError> {
        self.output = Some(tensor_descriptor_4d::<T>(n, c, h, w, format)?);
        Ok(self)
    }

    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn algorithms(mut self, requested_algo_count: usize) -> Self {
        let input = self.input.unwrap();
        let filter = self.filter.unwrap();
        let conv = self.conv.unwrap();
        let output = self.output.unwrap();

        self.fwd_algo =
            Some(convolution_algorithm(input, filter, conv, output, requested_algo_count).unwrap());
        self.bwd_data_algo = Some(
            convolution_backward_data_algorithm(input, filter, conv, output, requested_algo_count)
                .unwrap(),
        );
        self.bwd_filter_algo = Some(
            convolution_backward_filter_algorithm(
                input,
                filter,
                conv,
                output,
                requested_algo_count,
            )
            .unwrap(),
        );

        self
    }

    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn build(self) -> ConvolutionConfig<T> {
        let input = self.input.unwrap();
        let filter = self.filter.unwrap();
        let conv = self.conv.unwrap();
        let output = self.output.unwrap();
        let fwd_algo = self.fwd_algo.unwrap();
        let bwd_data_algo = self.bwd_data_algo.unwrap();
        let bwd_filter_algo = self.bwd_filter_algo.unwrap();

        let fwd_workspace = convolution_workspace(input, filter, conv, output, fwd_algo).unwrap();
        let bwd_data_workspace =
            convolution_backward_data_workspace(input, filter, conv, output, bwd_data_algo)
                .unwrap();
        let bwd_filter_workspace =
            convolution_backward_filter_workspace(input, filter, conv, output, bwd_filter_algo);

        ConvolutionConfig {
            input,
            filter,
            conv,
            output,
            fwd_algo,
            bwd_data_algo,
            bwd_filter_algo,
            fwd_workspace,
            bwd_data_workspace,
            bwd_filter_workspace,
            _marker: std::marker::PhantomData,
        }
    }
}

#[allow(clippy::missing_panics_doc)]
pub fn backward_bias<T: 'static + Copy>(
    alpha: T,
    d_output: *const T,
    beta: T,
    bias: *mut T,
    output_shape: &[usize],
) {
    let output_shape = output_shape
        .iter()
        .map(|x| i32::try_from(*x).unwrap())
        .collect::<Vec<i32>>();
    let output_shape_0 = output_shape[0];
    let output_shape_1 = output_shape[1];
    let output_shape_2 = output_shape[2];
    let output_shape_3 = output_shape[3];

    let state = ZENU_CUDA_STATE.lock().unwrap();
    let handle = state.get_cudnn();

    let output_desc = tensor_descriptor_4d::<T>(
        output_shape_0,
        output_shape_1,
        output_shape_2,
        output_shape_3,
        TensorFormat::NCHW,
    )
    .unwrap();

    let bias_desc = tensor_descriptor_4d::<T>(1, output_shape_1, 1, 1, TensorFormat::NCHW).unwrap();

    unsafe {
        cudnnConvolutionBackwardBias(
            handle.as_ptr(),
            std::ptr::from_ref(&alpha).cast(),
            output_desc,
            d_output.cast(),
            std::ptr::from_ref(&beta).cast(),
            bias_desc,
            bias.cast(),
        );
    }
}

#[cfg(test)]
#[allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::cast_ptr_alignment
)]
mod cudnn {
    use super::*;
    use crate::runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind};

    #[allow(clippy::similar_names)]
    #[test]
    fn random() {
        let input = [
            0.5432947,
            -0.39515755,
            0.20552567,
            -0.45032975,
            -0.5730771,
            -0.5553584,
            0.59432304,
            1.5419426,
            1.8197253,
            -0.5515287,
            -1.325326,
            0.18855357,
            -0.069072686,
            -0.49492535,
            -1.4959149,
            -0.19383712,
            -0.4731198,
            0.33555076,
            1.5091219,
            2.0819554,
            1.7067116,
            2.3803675,
            -1.1256016,
            -0.3169981,
            -0.14067143,
            0.8057536,
            0.3276143,
            -0.7607072,
            -1.599082,
            0.018486667,
            -0.7504268,
            0.18540798,
        ];
        let output = [
            0.3671525,
            -0.17387724,
            -0.53952014,
            -0.41356063,
            0.13519445,
            -0.6369239,
            -0.5777169,
            -0.07820636,
            -0.6019154,
            -0.85000455,
            -0.227178,
            0.38553098,
            0.53258127,
            0.4952766,
            0.16334829,
            0.5179188,
            -1.1829954,
            -0.15092221,
            0.15374796,
            0.5376092,
            -0.35269666,
            -0.10102463,
            -0.628401,
            -0.40036133,
            -0.5694187,
            -0.1765114,
            -0.05552435,
            -0.3107502,
            -0.6736164,
            -0.44401115,
            -0.1804393,
            0.056986123,
            0.5652461,
            0.8913239,
            0.30458608,
            -0.7666081,
            0.15480474,
            0.14275207,
            0.42336845,
            0.12534592,
            0.5706087,
            0.40240055,
            -0.16282544,
            -0.032061294,
            0.47645676,
            -0.09869753,
            -0.34638345,
            -0.02880986,
        ];
        let input_grad = [
            -0.06312838,
            0.05240719,
            0.05240719,
            0.21505278,
            -0.07415994,
            0.063570745,
            0.063570745,
            0.22900042,
            -0.07415994,
            0.063570745,
            0.063570745,
            0.22900042,
            -0.0014246926,
            0.13951382,
            0.13951382,
            0.005797662,
            -0.73124456,
            -0.7982433,
            -0.7982433,
            -0.098860174,
            -0.57463914,
            -0.689119,
            -0.689119,
            -0.12428501,
            -0.57463914,
            -0.689119,
            -0.689119,
            -0.12428501,
            -0.22594097,
            -0.37261552,
            -0.37261552,
            -0.085577406,
        ];
        let filter = [
            -0.0017646605,
            0.12644097,
            -0.1939936,
            -0.1734625,
            -0.090781756,
            0.063205294,
            -0.0046700113,
            0.18688585,
            -0.020917172,
            0.06236978,
            -0.071232304,
            -0.046330906,
            -0.2251778,
            -0.15610139,
            -0.09716192,
            0.008731253,
            0.0931814,
            0.14142673,
            -0.15979224,
            -0.10263957,
            0.0856111,
            0.19572432,
            -0.048507567,
            0.17637877,
            -0.03799128,
            0.024940623,
            0.21342279,
            -0.218654,
            -0.14838351,
            -0.05967162,
            -0.09187673,
            0.20364694,
            -0.1527774,
            -0.1085015,
            -0.16467114,
            -0.22074954,
            -0.13758895,
            0.2026092,
            0.105174676,
            0.11423842,
            0.01239595,
            -0.12084066,
            0.039877214,
            -0.22007395,
            -0.1703105,
            -0.121511586,
            0.1487135,
            0.13819724,
            -0.104532786,
            -0.0085047,
            0.1507459,
            0.23431942,
            0.093546025,
            0.03184169,
        ];
        let filter_grad = [
            -0.23757887,
            1.0425875,
            -0.7473556,
            -2.297492,
            -1.2111626,
            -2.932033,
            -2.651155,
            -1.1144958,
            -2.292071,
            5.325727,
            6.329977,
            5.2370563,
            2.994705,
            4.184363,
            4.690524,
            1.6231518,
            0.7308545,
            0.7638962,
            -0.23757887,
            1.0425875,
            -0.7473556,
            -2.297492,
            -1.2111626,
            -2.932033,
            -2.651155,
            -1.1144958,
            -2.292071,
            5.325727,
            6.329977,
            5.2370563,
            2.994705,
            4.184363,
            4.690524,
            1.6231518,
            0.7308545,
            0.7638962,
            -0.23757887,
            1.0425875,
            -0.7473556,
            -2.297492,
            -1.2111626,
            -2.932033,
            -2.651155,
            -1.1144958,
            -2.292071,
            5.325727,
            6.329977,
            5.2370563,
            2.994705,
            4.184363,
            4.690524,
            1.6231518,
            0.7308545,
            0.7638962,
        ];

        let input_gpu = cuda_malloc::<f32>(input.len()).unwrap();
        cuda_copy(
            input_gpu,
            input.as_ptr(),
            input.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        let output_gpu = cuda_malloc::<f32>(output.len()).unwrap();
        let input_grad_gpu = cuda_malloc::<f32>(input_grad.len()).unwrap();
        cuda_copy(
            input_grad_gpu,
            input_grad.as_ptr(),
            input_grad.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        let filter_gpu = cuda_malloc::<f32>(filter.len()).unwrap();
        cuda_copy(
            filter_gpu,
            filter.as_ptr(),
            filter.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        let filter_grad_gpu = cuda_malloc::<f32>(filter_grad.len()).unwrap();
        cuda_copy(
            filter_grad_gpu,
            filter_grad.as_ptr(),
            filter_grad.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        let conv_config = ConvolutionBuilder::default()
            .input(1, 2, 4, 4, TensorFormat::NCHW)
            .unwrap()
            .filter(3, 2, 3, 3, TensorFormat::NCHW)
            .unwrap()
            .conv(1, 1, 1, 1, 1, 1)
            .unwrap()
            .output(1, 3, 4, 4, TensorFormat::NCHW)
            .unwrap()
            .algorithms(1)
            .build();
        conv_config.forward(1.0, input_gpu, filter_gpu, 0.0, output_gpu);
        let mut output_cpu = vec![0.0; output.len()];
        cuda_copy(
            output_cpu.as_mut_ptr(),
            output_gpu,
            output.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();
        for (a, b) in output_cpu.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    #[allow(
        clippy::many_single_char_names,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::similar_names
    )]
    fn test_convolution_operations() {
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

        let conv_config = ConvolutionBuilder::default()
            .input(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .filter(k, c, kh, kw, TensorFormat::NCHW)
            .unwrap()
            .conv(pad_h, pad_w, stride_h, stride_w, 1, 1)
            .unwrap()
            .output(n, k, out_h, out_w, TensorFormat::NCHW)
            .unwrap()
            .algorithms(1)
            .build();

        // Forward pass test
        {
            // Create input tensor
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

            // Create filter tensor
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

            // Create output tensor
            let output_gpu = cuda_malloc::<f32>((n * k * out_h * out_w) as usize).unwrap();

            // Execute convolution
            conv_config.forward(1.0, input_gpu, filter_gpu, 0.0, output_gpu);

            // Copy output tensor to CPU
            let mut output_cpu = vec![0.0; (n * k * out_h * out_w) as usize];
            cuda_copy(
                output_cpu.as_mut_ptr(),
                output_gpu,
                (n * k * out_h * out_w) as usize,
                ZenuCudaMemCopyKind::DeviceToHost,
            )
            .unwrap();

            // Check output tensor
            let ans = vec![
                6888.0, 10218.0, 10479.0, 10740.0, 7056.0, 10296.0, 15219.0, 15570.0, 15921.0,
                10422.0, 11511.0, 16974.0, 17325.0, 17676.0, 11547.0, 12726.0, 18729.0, 19080.0,
                19431.0, 12672.0, 8040.0, 11784.0, 11991.0, 12198.0, 7920.0, 15960.0, 24069.0,
                24816.0, 25563.0, 17100.0, 25119.0, 37818.0, 38898.0, 39978.0, 26703.0, 28764.0,
                43218.0, 44298.0, 45378.0, 30258.0, 32409.0, 48618.0, 49698.0, 50778.0, 33813.0,
                21972.0, 32925.0, 33618.0, 34311.0, 22824.0, 25032.0, 37920.0, 39153.0, 40386.0,
                27144.0, 39942.0, 60417.0, 62226.0, 64035.0, 42984.0, 46017.0, 69462.0, 71271.0,
                73080.0, 48969.0, 52092.0, 78507.0, 80316.0, 82125.0, 54954.0, 35904.0, 54066.0,
                55245.0, 56424.0, 37728.0,
            ];
            assert_eq!(output_cpu, ans);
        }

        // Backward data test
        {
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

            conv_config.backward_data(1.0, filter_gpu, input_gpu, 0.0, output_gpu);

            let mut output_cpu = vec![0.0; (n * k * h * w) as usize];
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

        // Backward filter test
        {
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

            conv_config.backward_filter(1.0, input_gpu, output_gpu, 0.0, filter_gpu);

            let mut filter_cpu = vec![0.0; (k * c * kh * kw) as usize];
            cuda_copy(
                filter_cpu.as_mut_ptr(),
                filter_gpu,
                (k * c * kh * kw) as usize,
                ZenuCudaMemCopyKind::DeviceToHost,
            )
            .unwrap();

            let ans = vec![
                640.0, 770.0, 560.0, 1060.0, 1250.0, 900.0, 1240.0, 1470.0, 1080.0, 2640.0, 3020.0,
                2160.0, 3310.0, 3750.0, 2650.0, 3240.0, 3720.0, 2680.0, 4640.0, 5270.0, 3760.0,
                5560.0, 6250.0, 4400.0, 5240.0, 5970.0, 4280.0, 840.0, 1020.0, 760.0, 1290.0,
                1550.0, 1150.0, 1040.0, 1220.0, 880.0, 2840.0, 3270.0, 2360.0, 4040.0, 4675.0,
                3400.0, 3040.0, 3470.0, 2480.0, 4840.0, 5520.0, 3960.0, 6790.0, 7800.0, 5650.0,
                5040.0, 5720.0, 4080.0, 640.0, 770.0, 560.0, 1060.0, 1250.0, 900.0, 1240.0, 1470.0,
                1080.0, 2640.0, 3020.0, 2160.0, 3310.0, 3750.0, 2650.0, 3240.0, 3720.0, 2680.0,
                4640.0, 5270.0, 3760.0, 5560.0, 6250.0, 4400.0, 5240.0, 5970.0, 4280.0,
            ];

            assert_eq!(filter_cpu, ans);
        }
    }

    #[test]
    fn test_convolution_config_creation() {
        let conv_config = ConvolutionBuilder::<f32>::default()
            .input(2, 3, 5, 5, TensorFormat::NCHW)
            .unwrap()
            .filter(4, 3, 3, 3, TensorFormat::NCHW)
            .unwrap()
            .conv(1, 1, 1, 1, 1, 1)
            .unwrap()
            .output(2, 4, 5, 5, TensorFormat::NCHW)
            .unwrap()
            .algorithms(5)
            .build();

        // ここでは実際の操作は行わず、設定が正しく作成できたことを確認します
        assert!(
            conv_config.fwd_algo != cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_COUNT
        );
        assert!(
            conv_config.bwd_data_algo
                != cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT
        );
        assert!(
            conv_config.bwd_filter_algo
                != cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT
        );
    }
}

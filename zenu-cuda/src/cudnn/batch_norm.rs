use crate::ZENU_CUDA_STATE;

use zenu_cudnn_sys::*;

use super::{error::ZenuCudnnError, tensor_descriptor_4d, TensorFormat};

pub struct BatchNorm2d {
    input: cudnnTensorDescriptor_t,
    output: cudnnTensorDescriptor_t,
    scale_bias_mean_var: cudnnTensorDescriptor_t,
    mode: cudnnBatchNormMode_t,
}

pub struct BatchNorm2dBuilder {
    input: Option<cudnnTensorDescriptor_t>,
    output: Option<cudnnTensorDescriptor_t>,
    scale_bias_mean_var: Option<cudnnTensorDescriptor_t>,
    mode: Option<cudnnBatchNormMode_t>,
}

impl BatchNorm2d {
    pub fn forward_train<T: 'static>(
        &self,
        alpha: T,
        beta: T,
        x: *const std::ffi::c_void,
        y: *mut std::ffi::c_void,
        scale: *const std::ffi::c_void,
        bias: *const std::ffi::c_void,
        estimated_mean: *mut std::ffi::c_void,
        estimated_variance: *mut std::ffi::c_void,
        expotential_average_factor: f64,
        result_save_mean: *mut std::ffi::c_void,
        result_save_inv_variance: *mut std::ffi::c_void,
    ) -> Result<(), ZenuCudnnError> {
        let cudnn_handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();
        let status = unsafe {
            cudnnBatchNormalizationForwardTraining(
                cudnn_handle,
                self.mode,
                &alpha as *const T as *const std::ffi::c_void,
                &beta as *const T as *const std::ffi::c_void,
                self.input,
                x,
                self.output,
                y,
                self.scale_bias_mean_var,
                scale as *const T as *const std::ffi::c_void,
                bias,
                expotential_average_factor,
                estimated_mean,
                estimated_variance,
                1e-10,
                result_save_mean,
                result_save_inv_variance,
            )
        };
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        Ok(())
    }
}

impl Drop for BatchNorm2d {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyTensorDescriptor(self.input);
            cudnnDestroyTensorDescriptor(self.output);
            cudnnDestroyTensorDescriptor(self.scale_bias_mean_var);
        }
    }
}

impl BatchNorm2dBuilder {
    pub fn new() -> Self {
        Self {
            input: None,
            output: None,
            scale_bias_mean_var: None,
            mode: None,
        }
    }

    pub fn input(
        self,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        format: TensorFormat,
    ) -> Result<Self, ZenuCudnnError> {
        let input = tensor_descriptor_4d::<f32>(n, c, h, w, format)?;
        Ok(Self {
            input: Some(input),
            ..self
        })
    }

    pub fn output(
        self,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        format: TensorFormat,
    ) -> Result<Self, ZenuCudnnError> {
        let output = tensor_descriptor_4d::<f32>(n, c, h, w, format)?;
        Ok(Self {
            output: Some(output),
            ..self
        })
    }

    pub fn scale_bias_mean_var(
        self,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        format: TensorFormat,
    ) -> Result<Self, ZenuCudnnError> {
        let scale_bias_mean_var = tensor_descriptor_4d::<f32>(n, c, h, w, format)?;
        Ok(Self {
            scale_bias_mean_var: Some(scale_bias_mean_var),
            ..self
        })
    }

    pub fn mode(self, mode: cudnnBatchNormMode_t) -> Self {
        Self {
            mode: Some(mode),
            ..self
        }
    }

    pub fn build(self) -> Result<BatchNorm2d, ZenuCudnnError> {
        let input = self.input.expect("input is required");
        let output = self.output.expect("output is required");
        let scale_bias_mean_var = self
            .scale_bias_mean_var
            .expect("scale_bias_mean_var is required");
        let mode = self.mode.expect("mode is required");
        Ok(BatchNorm2d {
            input,
            output,
            scale_bias_mean_var,
            mode,
        })
    }
}

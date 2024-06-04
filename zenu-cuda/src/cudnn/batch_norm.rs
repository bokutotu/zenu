use crate::ZENU_CUDA_STATE;

use zenu_cudnn_sys::*;

use super::{error::ZenuCudnnError, tensor_descriptor_4d, TensorFormat};

pub struct BatchNorm2d<T> {
    input: cudnnTensorDescriptor_t,
    output: cudnnTensorDescriptor_t,
    scale_bias_mean_var: cudnnTensorDescriptor_t,
    mode: cudnnBatchNormMode_t,
    _phantom: std::marker::PhantomData<T>,
}

pub struct BatchNorm2dBuilder<T> {
    input: Option<cudnnTensorDescriptor_t>,
    output: Option<cudnnTensorDescriptor_t>,
    scale_bias_mean_var: Option<cudnnTensorDescriptor_t>,
    mode: Option<cudnnBatchNormMode_t>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: 'static> BatchNorm2d<T> {
    pub fn forward_train(
        &self,
        alpha: T,
        beta: T,
        x: *const T,
        y: *mut T,
        scale: *const T,
        bias: *const T,
        estimated_mean: *mut T,
        estimated_variance: *mut T,
        expotential_average_factor: f64,
        result_save_mean: *mut T,
        result_save_inv_variance: *mut T,
    ) -> Result<(), ZenuCudnnError> {
        let cudnn_handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();
        let status = unsafe {
            cudnnBatchNormalizationForwardTraining(
                cudnn_handle,
                self.mode,
                &alpha as *const T as *const std::ffi::c_void,
                &beta as *const T as *const std::ffi::c_void,
                self.input,
                x as *const std::ffi::c_void,
                self.output,
                y as *mut std::ffi::c_void,
                self.scale_bias_mean_var,
                scale as *const T as *const std::ffi::c_void,
                bias as *const T as *const std::ffi::c_void,
                expotential_average_factor,
                estimated_mean as *mut std::ffi::c_void,
                estimated_variance as *mut std::ffi::c_void,
                1e-10,
                result_save_mean as *mut std::ffi::c_void,
                result_save_inv_variance as *mut std::ffi::c_void,
            )
        };
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        Ok(())
    }
}

impl<T> Drop for BatchNorm2d<T> {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyTensorDescriptor(self.input);
            cudnnDestroyTensorDescriptor(self.output);
            cudnnDestroyTensorDescriptor(self.scale_bias_mean_var);
        }
    }
}

impl<T: 'static> BatchNorm2dBuilder<T> {
    pub fn new() -> Self {
        Self {
            input: None,
            output: None,
            scale_bias_mean_var: None,
            mode: None,
            _phantom: std::marker::PhantomData,
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
        let input = tensor_descriptor_4d::<T>(n, c, h, w, format)?;
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
        let output = tensor_descriptor_4d::<T>(n, c, h, w, format)?;
        Ok(Self {
            output: Some(output),
            ..self
        })
    }

    pub fn scale_bias_mean_var(self, c: i32, format: TensorFormat) -> Result<Self, ZenuCudnnError> {
        let scale_bias_mean_var = tensor_descriptor_4d::<T>(1, c, 1, 1, format)?;
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

    pub fn build(self) -> BatchNorm2d<T> {
        let input = self.input.expect("input is required");
        let output = self.output.expect("output is required");
        let scale_bias_mean_var = self
            .scale_bias_mean_var
            .expect("scale_bias_mean_var is required");
        let mode = self.mode.expect("mode is required");
        BatchNorm2d {
            input,
            output,
            scale_bias_mean_var,
            mode,
            _phantom: self._phantom,
        }
    }
}

pub struct BatchNorm2dBackward<T> {
    input: cudnnTensorDescriptor_t,
    input_grad: cudnnTensorDescriptor_t,
    output_grad: cudnnTensorDescriptor_t,
    scale_bias_mean_var: cudnnTensorDescriptor_t,
    mode: cudnnBatchNormMode_t,
    _phantom: std::marker::PhantomData<T>,
}

pub struct BatchNorm2dBackwardBuilder<T> {
    input: Option<cudnnTensorDescriptor_t>,
    input_grad: Option<cudnnTensorDescriptor_t>,
    output_grad: Option<cudnnTensorDescriptor_t>,
    scale_bias_mean_var: Option<cudnnTensorDescriptor_t>,
    mode: Option<cudnnBatchNormMode_t>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: 'static> BatchNorm2dBackward<T> {
    pub fn backward(
        &self,
        alpha_data_grad: T,
        beta_data_grad: T,
        alpha_param_diff: T,
        beta_param_diff: T,
        x: *const T,
        y_grad: *mut T,
        x_grad: *mut T,
        scale: *const T,
        scale_grad: *mut T,
        bias_grad: *mut T,
        result_save_mean: *const T,
        result_save_inv_variance: *const T,
    ) -> Result<(), ZenuCudnnError> {
        let cudnn_handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();
        let status = unsafe {
            cudnnBatchNormalizationBackward(
                cudnn_handle,
                self.mode,
                &alpha_data_grad as *const T as *const std::ffi::c_void,
                &beta_data_grad as *const T as *const std::ffi::c_void,
                &alpha_param_diff as *const T as *const std::ffi::c_void,
                &beta_param_diff as *const T as *const std::ffi::c_void,
                self.input,
                x as *const std::ffi::c_void,
                self.output_grad,
                y_grad as *const std::ffi::c_void,
                self.input_grad,
                x_grad as *mut std::ffi::c_void,
                self.scale_bias_mean_var,
                scale as *const std::ffi::c_void,
                scale_grad as *mut std::ffi::c_void,
                bias_grad as *mut std::ffi::c_void,
                1e-10,
                result_save_mean as *const std::ffi::c_void,
                result_save_inv_variance as *const std::ffi::c_void,
            )
        };
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        Ok(())
    }
}

impl<T> Drop for BatchNorm2dBackward<T> {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyTensorDescriptor(self.input);
            cudnnDestroyTensorDescriptor(self.input_grad);
            cudnnDestroyTensorDescriptor(self.output_grad);
            cudnnDestroyTensorDescriptor(self.scale_bias_mean_var);
        }
    }
}

impl<T: 'static> BatchNorm2dBackwardBuilder<T> {
    pub fn new() -> Self {
        Self {
            input: None,
            input_grad: None,
            output_grad: None,
            scale_bias_mean_var: None,
            mode: None,
            _phantom: std::marker::PhantomData,
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
        let input = tensor_descriptor_4d::<T>(n, c, h, w, format)?;
        Ok(Self {
            input: Some(input),
            ..self
        })
    }

    pub fn input_grad(
        self,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        format: TensorFormat,
    ) -> Result<Self, ZenuCudnnError> {
        let input_grad = tensor_descriptor_4d::<T>(n, c, h, w, format)?;
        Ok(Self {
            input_grad: Some(input_grad),
            ..self
        })
    }

    pub fn output_grad(
        self,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        format: TensorFormat,
    ) -> Result<Self, ZenuCudnnError> {
        let output_grad = tensor_descriptor_4d::<T>(n, c, h, w, format)?;
        Ok(Self {
            output_grad: Some(output_grad),
            ..self
        })
    }

    pub fn scale_bias_mean_var(self, c: i32, format: TensorFormat) -> Result<Self, ZenuCudnnError> {
        let scale_bias_mean_var = tensor_descriptor_4d::<T>(1, c, 1, 1, format)?;
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

    pub fn build(self) -> BatchNorm2dBackward<T> {
        let input = self.input.expect("input is required");
        let input_grad = self.input_grad.expect("input_grad is required");
        let output_grad = self.output_grad.expect("output_grad is required");
        let scale_bias_mean_var = self
            .scale_bias_mean_var
            .expect("scale_bias_mean_var is required");
        let mode = self.mode.expect("mode is required");
        BatchNorm2dBackward {
            input,
            input_grad,
            output_grad,
            scale_bias_mean_var,
            mode,
            _phantom: self._phantom,
        }
    }
}

pub struct BatchNorm2dInference<T> {
    input: cudnnTensorDescriptor_t,
    output: cudnnTensorDescriptor_t,
    scale_bias_mean_var: cudnnTensorDescriptor_t,
    mode: cudnnBatchNormMode_t,
    _phantom: std::marker::PhantomData<T>,
}

pub struct BatchNorm2dInferenceBuilder<T> {
    input: Option<cudnnTensorDescriptor_t>,
    output: Option<cudnnTensorDescriptor_t>,
    scale_bias_mean_var: Option<cudnnTensorDescriptor_t>,
    mode: Option<cudnnBatchNormMode_t>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: 'static> BatchNorm2dInference<T> {
    pub fn forward_inference(
        &self,
        alpha: T,
        beta: T,
        x: *const T,
        y: *mut T,
        scale: *const T,
        bias: *const T,
        estimated_mean: *const T,
        estimated_variance: *const T,
    ) -> Result<(), ZenuCudnnError> {
        let cudnn_handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();
        let status = unsafe {
            cudnnBatchNormalizationForwardInference(
                cudnn_handle,
                self.mode,
                &alpha as *const T as *const std::ffi::c_void,
                &beta as *const T as *const std::ffi::c_void,
                self.input,
                x as *const std::ffi::c_void,
                self.output,
                y as *mut std::ffi::c_void,
                self.scale_bias_mean_var,
                scale as *const T as *const std::ffi::c_void,
                bias as *const T as *const std::ffi::c_void,
                estimated_mean as *const std::ffi::c_void,
                estimated_variance as *const std::ffi::c_void,
                1e-10,
            )
        };
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        Ok(())
    }
}

impl<T> Drop for BatchNorm2dInference<T> {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyTensorDescriptor(self.input);
            cudnnDestroyTensorDescriptor(self.output);
            cudnnDestroyTensorDescriptor(self.scale_bias_mean_var);
        }
    }
}

impl<T: 'static> BatchNorm2dInferenceBuilder<T> {
    pub fn new() -> Self {
        Self {
            input: None,
            output: None,
            scale_bias_mean_var: None,
            mode: None,
            _phantom: std::marker::PhantomData,
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
        let input = tensor_descriptor_4d::<T>(n, c, h, w, format)?;
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
        let output = tensor_descriptor_4d::<T>(n, c, h, w, format)?;
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
        let scale_bias_mean_var = tensor_descriptor_4d::<T>(n, c, h, w, format)?;
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

    pub fn build(self) -> BatchNorm2dInference<T> {
        let input = self.input.expect("input is required");
        let output = self.output.expect("output is required");
        let scale_bias_mean_var = self
            .scale_bias_mean_var
            .expect("scale_bias_mean_var is required");
        let mode = self.mode.expect("mode is required");
        BatchNorm2dInference {
            input,
            output,
            scale_bias_mean_var,
            mode,
            _phantom: self._phantom,
        }
    }
}

#[cfg(test)]
mod batch_norm {
    use zenu_cudnn_sys::cudnnBatchNormMode_t;

    use crate::{
        cudnn::TensorFormat,
        runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind},
    };

    use super::BatchNorm2dBuilder;

    #[test]
    fn forward() {
        let n = 2;
        let c = 3;
        let h = 4;
        let w = 4;
        let x_cpu = vec![
            -1.1258398,
            -1.1523602,
            -0.25057858,
            -0.4338788,
            0.84871036,
            0.69200915,
            -0.31601277,
            -2.1152194,
            0.32227492,
            -1.2633348,
            0.3499832,
            0.30813393,
            0.11984151,
            1.2376579,
            1.1167772,
            -0.24727815,
            -1.3526537,
            -1.6959312,
            0.5666506,
            0.79350835,
            0.59883946,
            -1.5550951,
            -0.3413604,
            1.8530061,
            0.7501895,
            -0.58549756,
            -0.17339675,
            0.18347794,
            1.3893661,
            1.5863342,
            0.94629836,
            -0.84367675,
            -0.6135831,
            0.03159274,
            -0.49267697,
            0.24841475,
            0.43969584,
            0.112411186,
            0.64079237,
            0.44115627,
            -0.10230965,
            0.792444,
            -0.2896677,
            0.052507486,
            0.52286047,
            2.3022053,
            -1.4688939,
            -1.5866888,
            -0.6730899,
            0.8728312,
            1.0553575,
            0.17784372,
            -0.23033547,
            -0.3917544,
            0.5432947,
            -0.39515755,
            -0.44621718,
            0.7440207,
            1.5209795,
            3.4105027,
            -1.5311843,
            -1.234135,
            1.8197253,
            -0.5515287,
            -0.5692481,
            0.9199714,
            1.1108161,
            1.2898741,
            -1.478174,
            2.5672328,
            -0.4731198,
            0.33555076,
            -1.629326,
            -0.54974365,
            -0.47983426,
            -0.49968153,
            -1.0669804,
            1.1149396,
            -0.14067143,
            0.8057536,
            -0.093348235,
            0.6870502,
            -0.83831537,
            0.00089182175,
            0.8418941,
            -0.40003416,
            1.039462,
            0.3581531,
            -0.24600095,
            2.3025165,
            -1.8816892,
            -0.049727023,
            -1.0449786,
            -0.9565008,
            0.03353186,
            0.7100866,
        ];
        let scale = vec![2.0575912, -0.03542188, 0.06271883];
        let bias = vec![-0.7663063, 1.0992506, 2.7565384];
        let mean = vec![0.0, 0.0, 0.0];
        let variance = vec![1.0, 1.0, 1.0];

        let x_gpu = cuda_malloc(x_cpu.len()).unwrap();
        let y_gpu = cuda_malloc(x_cpu.len()).unwrap();
        let scale_gpu = cuda_malloc(scale.len()).unwrap();
        let bias_gpu = cuda_malloc(bias.len()).unwrap();
        let mean_gpu = cuda_malloc(mean.len()).unwrap();
        let variance_gpu = cuda_malloc(variance.len()).unwrap();

        cuda_copy(
            x_gpu,
            x_cpu.as_ptr(),
            x_cpu.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        cuda_copy(
            scale_gpu,
            scale.as_ptr(),
            scale.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        cuda_copy(
            bias_gpu,
            bias.as_ptr(),
            bias.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        cuda_copy(
            mean_gpu,
            mean.as_ptr(),
            mean.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        cuda_copy(
            variance_gpu,
            variance.as_ptr(),
            variance.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        let batch_norm = BatchNorm2dBuilder::<f64>::new()
            .input(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .output(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .scale_bias_mean_var(c, TensorFormat::NCHW)
            .unwrap()
            .mode(cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL)
            .build();

        let alpha = 1.0;
        let beta = 0.0;
        let average_factor = 0.1;

        let result_saveing_mean = cuda_malloc(mean.len()).unwrap();
        let result_saveing_inv_variance = cuda_malloc(variance.len()).unwrap();

        batch_norm
            .forward_train(
                alpha,
                beta,
                x_gpu,
                y_gpu,
                scale_gpu,
                bias_gpu,
                mean_gpu,
                variance_gpu,
                average_factor,
                result_saveing_mean,
                result_saveing_inv_variance,
            )
            .unwrap();

        let mut y_cpu = vec![0.0; x_cpu.len()];
        cuda_copy(
            y_cpu.as_mut_ptr(),
            y_gpu,
            y_cpu.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();
        let y_ans = vec![
            -3.0458143,
            -3.0956612,
            -1.4006953,
            -1.7452217,
            0.6655005,
            0.37096885,
            -1.5236837,
            -4.905427,
            -0.32397428,
            -3.3042462,
            -0.2718945,
            -0.35055333,
            -0.704463,
            1.3965565,
            1.1693522,
            -1.3944919,
            1.1464527,
            1.1575645,
            1.0843246,
            1.0769812,
            1.0832826,
            1.1530057,
            1.113717,
            1.0426852,
            1.0783834,
            1.1216197,
            1.10828,
            1.0967278,
            1.0576931,
            1.0513173,
            1.0720353,
            1.129977,
            2.7115202,
            2.7555108,
            2.719764,
            2.7702947,
            2.7833369,
            2.7610214,
            2.7970483,
            2.7834363,
            2.7463808,
            2.8073885,
            2.733606,
            2.7569368,
            2.7890072,
            2.9103298,
            2.653202,
            2.6451702,
            -2.1948369,
            0.7108375,
            1.0539092,
            -0.5954435,
            -1.3626468,
            -1.6660458,
            0.091448985,
            -1.6724422,
            -1.7684126,
            0.46872845,
            1.9290806,
            5.480581,
            -3.8076894,
            -3.2493632,
            2.4905956,
            -1.9663535,
            1.1210938,
            1.0728875,
            1.0667099,
            1.0609138,
            1.1505157,
            1.0195656,
            1.117982,
            1.0918053,
            1.1554085,
            1.1204623,
            1.1181993,
            1.1188419,
            1.1372054,
            1.0665764,
            1.1072206,
            1.0765848,
            2.7469919,
            2.8002024,
            2.6961973,
            2.7534175,
            2.8107603,
            2.726081,
            2.8242311,
            2.777777,
            2.7365835,
            2.910351,
            2.625056,
            2.7499661,
            2.682106,
            2.688139,
            2.7556431,
            2.801773,
        ];
        for i in 0..y_cpu.len() {
            let diff: f64 = y_cpu[i] - y_ans[i];
            let diff_abs = diff.abs();
            assert!(diff_abs.abs() < 1e-6);
        }

        let y_grad_cpu = vec![1.0; y_cpu.len()];
        let y_grad_gpu = cuda_malloc(y_grad_cpu.len()).unwrap();
        cuda_copy(
            y_grad_gpu,
            y_grad_cpu.as_ptr(),
            y_grad_cpu.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        let x_grad_gpu = cuda_malloc(x_cpu.len()).unwrap();
        let scale_grad_gpu = cuda_malloc(scale.len()).unwrap();
        let bias_grad_gpu = cuda_malloc(bias.len()).unwrap();

        let batch_norm_backward = super::BatchNorm2dBackwardBuilder::<f64>::new()
            .input(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .input_grad(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .output_grad(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .scale_bias_mean_var(c, TensorFormat::NCHW)
            .unwrap()
            .mode(cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL)
            .build();
        batch_norm_backward
            .backward(
                alpha,
                beta,
                alpha,
                beta,
                y_gpu,
                y_grad_gpu,
                x_grad_gpu,
                scale_gpu,
                scale_grad_gpu,
                bias_grad_gpu,
                // mean_gpu,
                // variance_gpu,
                std::ptr::null(),
                std::ptr::null(),
            )
            .unwrap();

        let mut x_grad_cpu = vec![0.0; x_cpu.len()];
        cuda_copy(
            x_grad_cpu.as_mut_ptr(),
            x_grad_gpu,
            x_grad_cpu.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();
        let mut scale_grad_cpu = vec![0.0; scale.len()];
        cuda_copy(
            scale_grad_cpu.as_mut_ptr(),
            scale_grad_gpu,
            scale_grad_cpu.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();
        let mut bias_grad_cpu = vec![0.0; bias.len()];
        cuda_copy(
            bias_grad_cpu.as_mut_ptr(),
            bias_grad_gpu,
            bias_grad_cpu.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();

        println!("{:?}", x_grad_cpu);
        println!("{:?}", scale_grad_cpu);
        println!("{:?}", bias_grad_cpu);
        let x_grad_ans = vec![
            1.4172037e-08,
            1.4481942e-08,
            3.9440895e-09,
            6.086061e-09,
            -8.901752e-09,
            -7.0706063e-09,
            4.7087267e-09,
            2.573352e-08,
            -2.7500429e-09,
            1.5778745e-08,
            -3.0738305e-09,
            -2.584797e-09,
            -3.8448877e-10,
            -1.3446835e-08,
            -1.2034271e-08,
            3.9055217e-09,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -7.928192e-10,
            -1.8097595e-11,
            -6.476361e-10,
            2.422604e-10,
            4.719491e-10,
            7.89485e-11,
            7.1342404e-10,
            4.737028e-10,
            -1.7888643e-10,
            8.9552604e-10,
            -4.0386436e-10,
            7.016652e-12,
            5.718125e-10,
            2.7084346e-09,
            -1.819869e-09,
            -1.961316e-09,
            8.8813845e-09,
            -9.18362e-09,
            -1.1316547e-08,
            -1.0622789e-09,
            3.7075367e-09,
            5.5938125e-09,
            -5.3327907e-09,
            5.6335803e-09,
            6.230242e-09,
            -7.678392e-09,
            -1.6757618e-08,
            -3.8837815e-08,
            1.8908725e-08,
            1.5437529e-08,
            -2.0248637e-08,
            7.460869e-09,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -1.6812565e-10,
            7.6897017e-10,
            -1.0626757e-09,
            -5.4962992e-11,
            9.549054e-10,
            -5.363915e-10,
            1.1921432e-09,
            3.740333e-10,
            -3.514297e-10,
            2.708808e-09,
            -2.31555e-09,
            -1.1574567e-10,
            -1.3108351e-09,
            -1.204592e-09,
            -1.5769119e-11,
            7.96632e-10,
        ];
        for i in 0..x_grad_cpu.len() {
            let diff: f64 = x_grad_cpu[i] - x_grad_ans[i];
            let diff_abs = diff.abs();
            assert!(diff_abs.abs() < 1e-6);
        }

        let scale_grad_ans = vec![2.1779134e-07, 0.0, -5.18386e-07];
        let bias_grad_ans = vec![32.0, 32.0, 32.0];

        for i in 0..scale_grad_cpu.len() {
            let diff: f64 = scale_grad_cpu[i] - scale_grad_ans[i];
            let diff_abs = diff.abs();
            assert!(diff_abs.abs() < 1e-6);
        }

        println!("{:?}", bias_grad_cpu);
        for i in 0..bias_grad_cpu.len() {
            let diff: f64 = bias_grad_cpu[i] - bias_grad_ans[i];
            let diff_abs = diff.abs();
            assert!(diff_abs.abs() < 1e-6);
        }
    }
}

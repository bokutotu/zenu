use zenu_cudnn_sys::{
    cudnnBatchNormMode_t, cudnnBatchNormalizationBackward, cudnnBatchNormalizationForwardInference,
    cudnnBatchNormalizationForwardTraining, cudnnDestroyTensorDescriptor, cudnnStatus_t,
    cudnnTensorDescriptor_t,
};

use crate::ZENU_CUDA_STATE;

use super::{error::ZenuCudnnError, tensor_descriptor_4d, TensorFormat};

#[allow(clippy::module_name_repetitions)]
pub struct BatchNorm2d<T> {
    input: cudnnTensorDescriptor_t,
    output: cudnnTensorDescriptor_t,
    scale_bias_mean_var: cudnnTensorDescriptor_t,
    mode: cudnnBatchNormMode_t,
    _phantom: std::marker::PhantomData<T>,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Default)]
pub struct BatchNorm2dBuilder<T> {
    input: Option<cudnnTensorDescriptor_t>,
    output: Option<cudnnTensorDescriptor_t>,
    scale_bias_mean_var: Option<cudnnTensorDescriptor_t>,
    mode: Option<cudnnBatchNormMode_t>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: 'static + Copy> BatchNorm2d<T> {
    pub fn new(
        input: cudnnTensorDescriptor_t,
        output: cudnnTensorDescriptor_t,
        scale_bias_mean_var: cudnnTensorDescriptor_t,
        mode: cudnnBatchNormMode_t,
    ) -> Self {
        BatchNorm2d {
            input,
            output,
            scale_bias_mean_var,
            mode,
            _phantom: std::marker::PhantomData,
        }
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::missing_errors_doc,
        clippy::missing_panics_doc
    )]
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
                std::ptr::from_ref(&alpha).cast(),
                std::ptr::from_ref(&beta).cast(),
                self.input,
                x.cast(),
                self.output,
                y.cast(),
                self.scale_bias_mean_var,
                scale.cast(),
                bias.cast(),
                expotential_average_factor,
                estimated_mean.cast(),
                estimated_variance.cast(),
                1e-10,
                result_save_mean.cast(),
                result_save_inv_variance.cast(),
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
    #[must_use]
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

    #[must_use]
    pub fn mode(self, mode: cudnnBatchNormMode_t) -> Self {
        Self {
            mode: Some(mode),
            ..self
        }
    }

    #[must_use]
    pub fn build(self) -> BatchNorm2d<T> {
        let input = self.input.expect("input is required");
        let output = self.output.expect("output is required");
        let scale_bias_mean_var = self
            .scale_bias_mean_var
            .expect("scale_bias_mean_var is required");
        // let mode = self.mode.expect("mode is required");
        let mode = self
            .mode
            .unwrap_or(cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL);
        BatchNorm2d {
            input,
            output,
            scale_bias_mean_var,
            mode,
            _phantom: self._phantom,
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct BatchNorm2dBackward<T> {
    input: cudnnTensorDescriptor_t,
    input_grad: cudnnTensorDescriptor_t,
    output_grad: cudnnTensorDescriptor_t,
    scale_bias_mean_var: cudnnTensorDescriptor_t,
    mode: cudnnBatchNormMode_t,
    _phantom: std::marker::PhantomData<T>,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Default)]
pub struct BatchNorm2dBackwardBuilder<T> {
    input: Option<cudnnTensorDescriptor_t>,
    input_grad: Option<cudnnTensorDescriptor_t>,
    output_grad: Option<cudnnTensorDescriptor_t>,
    scale_bias_mean_var: Option<cudnnTensorDescriptor_t>,
    mode: Option<cudnnBatchNormMode_t>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: 'static + Copy> BatchNorm2dBackward<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        alpha_data_grad: T,
        beta_data_grad: T,
        alpha_param_diff: T,
        beta_param_diff: T,
        x: *const T,
        y_grad: *const T,
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
                std::ptr::from_ref(&alpha_data_grad).cast(),
                std::ptr::from_ref(&beta_data_grad).cast(),
                std::ptr::from_ref(&alpha_param_diff).cast(),
                std::ptr::from_ref(&beta_param_diff).cast(),
                self.input,
                x.cast(),
                self.output_grad,
                y_grad.cast(),
                self.input_grad,
                x_grad.cast(),
                self.scale_bias_mean_var,
                scale.cast(),
                scale_grad.cast(),
                bias_grad.cast(),
                1e-10,
                result_save_mean.cast(),
                result_save_inv_variance.cast(),
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
    #[must_use]
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

    #[must_use]
    pub fn mode(self, mode: cudnnBatchNormMode_t) -> Self {
        Self {
            mode: Some(mode),
            ..self
        }
    }

    #[must_use]
    pub fn build(self) -> BatchNorm2dBackward<T> {
        let input = self.input.expect("input is required");
        let input_grad = self.input_grad.expect("input_grad is required");
        let output_grad = self.output_grad.expect("output_grad is required");
        let scale_bias_mean_var = self
            .scale_bias_mean_var
            .expect("scale_bias_mean_var is required");
        let mode = self
            .mode
            .unwrap_or(cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL);
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

#[allow(clippy::module_name_repetitions)]
pub struct BatchNorm2dInference<T> {
    input: cudnnTensorDescriptor_t,
    output: cudnnTensorDescriptor_t,
    scale_bias_mean_var: cudnnTensorDescriptor_t,
    mode: cudnnBatchNormMode_t,
    _phantom: std::marker::PhantomData<T>,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Default)]
pub struct BatchNorm2dInferenceBuilder<T> {
    input: Option<cudnnTensorDescriptor_t>,
    output: Option<cudnnTensorDescriptor_t>,
    scale_bias_mean_var: Option<cudnnTensorDescriptor_t>,
    mode: Option<cudnnBatchNormMode_t>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: 'static + Copy> BatchNorm2dInference<T> {
    #[allow(clippy::too_many_arguments)]
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
                std::ptr::from_ref(&alpha).cast(),
                std::ptr::from_ref(&beta).cast(),
                self.input,
                x.cast(),
                self.output,
                y.cast(),
                self.scale_bias_mean_var,
                scale.cast(),
                bias.cast(),
                estimated_mean.cast(),
                estimated_variance.cast(),
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
    #[must_use]
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

    #[must_use]
    pub fn mode(self, mode: cudnnBatchNormMode_t) -> Self {
        Self {
            mode: Some(mode),
            ..self
        }
    }

    #[must_use]
    pub fn build(self) -> BatchNorm2dInference<T> {
        let input = self.input.expect("input is required");
        let output = self.output.expect("output is required");
        let scale_bias_mean_var = self
            .scale_bias_mean_var
            .expect("scale_bias_mean_var is required");
        let mode = self
            .mode
            .unwrap_or(cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL);
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
mod batch_norm_test {
    use zenu_cudnn_sys::cudnnBatchNormMode_t;

    use crate::{
        cudnn::{batch_norm::BatchNorm2dBackwardBuilder, TensorFormat},
        runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind},
    };

    use super::BatchNorm2dBuilder;

    fn cpu_vec_to_gpu<T: 'static>(vec: &[T]) -> *mut T {
        let gpu = cuda_malloc(vec.len()).unwrap();
        cuda_copy(
            gpu,
            vec.as_ptr(),
            vec.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        gpu
    }

    fn gpu_to_cpu_vec<T: 'static + Default + Clone>(gpu: *const T, len: usize) -> Vec<T> {
        let mut vec = vec![T::default(); len];
        cuda_copy(
            vec.as_mut_ptr(),
            gpu,
            len,
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();
        vec
    }

    #[test]
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn forward() {
        // import torch
        // import torch.nn as nn
        // import numpy as np
        //
        // # シードの固定
        // torch.manual_seed(0)
        // np.random.seed(0)
        //
        // # デバイスを設定
        // device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        //
        // # バッチ正規化レイヤーを定義
        // class BatchNormModel(nn.Module):
        //     def __init__(self, num_features):
        //         super(BatchNormModel, self).__init__()
        //         self.bn = nn.BatchNorm2d(num_features)
        //
        //     def forward(self, x):
        //         return self.bn(x)
        //
        // # 入力データの設定
        // input_data = torch.randn(2, 2, 2, 2).to(device)  # バッチサイズ 2, チャンネル 2, 高さ 2, 幅 2
        // input_data.requires_grad = True
        //
        // # PyTorchモデルのインスタンスを作成
        // model = BatchNormModel(num_features=2).to(device)
        //
        // # フォワードパス
        // output = model(input_data)
        //
        // # 入力データの出力を追加
        // print("Input Data:\n", input_data.cpu().detach().numpy())
        //
        // # バッチ正規化の内部パラメータを取得
        // running_mean = model.bn.running_mean  # resultRunningMean に対応
        // running_var = model.bn.running_var    # resultRunningVariance に対応
        //
        // # 現在のバッチ用のパラメータ
        // saved_mean = model.bn.running_mean.clone().detach()          # resultSaveMean に対応
        // saved_var = (model.bn.running_var.clone().detach().reciprocal())  # resultSaveInvVariance に対応
        //
        // # スケールとバイアスの取得
        // scale = model.bn.weight.clone().detach()  # bnScale に対応
        // bias = model.bn.bias.clone().detach()     # bnBias に対応
        //
        // print("Running Mean:\n", running_mean.cpu().numpy())
        // print("Running Variance:\n", running_var.cpu().numpy())
        // print("Saved Mean (for current batch):\n", saved_mean.cpu().numpy())
        // print("Saved Inv Variance (for current batch):\n", saved_var.cpu().numpy())
        // print("Scale (bnScale):\n", scale.cpu().numpy())
        // print("Bias (bnBias):\n", bias.cpu().numpy())
        //
        // # ランダムな出力に対する勾配を生成
        // output_grad = torch.randn_like(output).to(device)
        //
        // # 出力に対するダミーの損失を計算し、バックワードパスを実行
        // output.backward(gradient=output_grad)
        //
        // # 勾配を取得
        // grad_input = input_data.grad   # dInput に対応
        // grad_bn_weight = model.bn.weight.grad  # dBnScale に対応
        // grad_bn_bias = model.bn.bias.grad      # dBnBias に対応
        //
        // # 出力用に追加された部分
        // print("Output Gradient:\n", output_grad.cpu().numpy())
        //
        // print("Gradient w.r.t input:\n", grad_input.cpu().numpy())
        // print("Gradient w.r.t bn_weight (Scale):\n", grad_bn_weight.cpu().numpy())
        // print("Gradient w.r.t bn_bias:\n", grad_bn_bias.cpu().numpy())
        let n = 2;
        let c = 2;
        let h = 2;
        let w = 2;
        let batch_norm = BatchNorm2dBuilder::new()
            .mode(cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL)
            .input(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .output(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .scale_bias_mean_var(c, TensorFormat::NCHW)
            .unwrap()
            .build();

        let batch_norm_backward = BatchNorm2dBackwardBuilder::new()
            .mode(cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL)
            .input(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .input_grad(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .output_grad(n, c, h, w, TensorFormat::NCHW)
            .unwrap()
            .scale_bias_mean_var(c, TensorFormat::NCHW)
            .unwrap()
            .build();

        let input_cpu = [
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
        ];
        let output_cpu = [
            -1.0970649,
            -1.1374662,
            0.23631285,
            -0.04292771,
            0.66504365,
            0.5121599,
            -0.4713051,
            -2.2266803,
            1.109001,
            -1.3065253,
            1.1512119,
            1.0874585,
            -0.04606889,
            1.0445158,
            0.92657995,
            -0.40424496,
        ];
        let running_mean = [-0.04057, 0.01670607];
        let running_variance = [0.9492437, 1.0200632];
        let saved_mean = [-0.04057, 0.01670607];
        let saved_variance = [0.9492437, 1.0200632];
        let scale = [1.0, 1.0];
        let bias = [0.0, 0.0];

        let input_gpu = cpu_vec_to_gpu(&input_cpu);
        let running_mean_gpu = cpu_vec_to_gpu(&running_mean);
        let running_variance_gpu = cpu_vec_to_gpu(&running_variance);
        let saved_mean_gpu = cuda_malloc(saved_mean.len()).unwrap();
        let saved_variance_gpu = cuda_malloc(saved_variance.len()).unwrap();

        let output_gpu = cuda_malloc(input_cpu.len()).unwrap();
        let scale_gpu = cpu_vec_to_gpu(&scale);
        let bias_gpu = cpu_vec_to_gpu(&bias);

        batch_norm
            .forward_train(
                1.0,
                0.0,
                input_gpu,
                output_gpu,
                scale_gpu,
                bias_gpu,
                running_mean_gpu,
                running_variance_gpu,
                1.0,
                saved_mean_gpu,
                saved_variance_gpu,
            )
            .unwrap();
        let output_result = gpu_to_cpu_vec(output_gpu, input_cpu.len());
        for i in 0..output_cpu.len() {
            assert!(((output_cpu[i] - output_result[i]) as f64).abs() < 1e-4);
        }
        let output_grad_cpu = vec![
            -0.9246624,
            -0.42534423,
            -2.6438458,
            0.14518386,
            -0.1208664,
            -0.57972574,
            -0.622851,
            -0.3283869,
            -1.0745419,
            -0.36314395,
            -1.6710504,
            2.2655048,
            0.3116848,
            -0.1841891,
            1.2866427,
            1.1819527,
        ];
        let output_grad: *mut f64 = cpu_vec_to_gpu(&output_grad_cpu);
        let input_grad = cuda_malloc(input_cpu.len()).unwrap();
        let scale_grad = cuda_malloc(scale.len()).unwrap();
        let bias_grad = cuda_malloc(bias.len()).unwrap();

        batch_norm_backward
            .backward(
                1.0,
                0.,
                1.,
                0.,
                input_gpu,
                output_grad,
                input_grad,
                scale_gpu,
                scale_grad,
                bias_grad,
                saved_mean_gpu,
                saved_variance_gpu,
            )
            .unwrap();

        let input_grad_cpu = gpu_to_cpu_vec(input_grad, input_cpu.len());
        let input_grad_ans = [
            -0.37104672,
            0.3949252,
            -3.165237,
            1.1202719,
            -0.32676408,
            -0.7529081,
            -0.6564417,
            -0.12187076,
            -0.8892036,
            0.5118922,
            -1.8034735,
            4.201872,
            0.19542423,
            -0.44200054,
            1.0096132,
            1.0949475,
        ];
        for i in 0..input_grad_cpu.len() {
            assert!(((input_grad_cpu[i] - input_grad_ans[i]) as f64).abs() < 1e-4);
        }
    }
}

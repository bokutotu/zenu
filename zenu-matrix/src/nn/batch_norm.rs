use crate::{
    device::{cpu::Cpu, Device, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Ref},
    num::Num,
};

#[cfg(feature = "nvidia")]
use zenu_cuda::cudnn::{batch_norm::*, TensorFormat};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

pub struct BatchNorm2dConfig<T: Num> {
    #[cfg(feature = "nvidia")]
    pub device_batch_norm: BatchNorm2d<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Num> BatchNorm2dConfig<T> {
    pub fn new(dim: DimDyn) -> Self {
        BatchNorm2dConfig::<T> {
            #[cfg(feature = "nvidia")]
            device_batch_norm: create_batch_norm_gpu::<T>(dim),
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct BatchNorm2dBackwardConfig<T> {
    #[cfg(feature = "nvidia")]
    pub device_batch_norm_backward: BatchNorm2dBackward<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Num> BatchNorm2dBackwardConfig<T> {
    pub fn new(dim: DimDyn) -> Self {
        BatchNorm2dBackwardConfig::<T> {
            #[cfg(feature = "nvidia")]
            device_batch_norm_backward: create_batch_norm_backward_gpu::<T>(dim),
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct BatchNorm2dInferenceConfig<T> {
    #[cfg(feature = "nvidia")]
    pub device_batch_norm_inference: BatchNorm2dInference<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Num> BatchNorm2dInferenceConfig<T> {
    pub fn new(dim: DimDyn) -> Self {
        BatchNorm2dInferenceConfig::<T> {
            #[cfg(feature = "nvidia")]
            device_batch_norm_inference: create_batch_norm_inference_gpu::<T>(dim),
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "nvidia")]
fn create_batch_norm_gpu<T: Num>(input: DimDyn) -> BatchNorm2d<T> {
    let input = (
        input[0].try_into().unwrap(),
        input[1].try_into().unwrap(),
        input[2].try_into().unwrap(),
        input[3].try_into().unwrap(),
    );
    let batch_norm = BatchNorm2dBuilder::<T>::new()
        .input(input.0, input.1, input.2, input.3, TensorFormat::NCHW)
        .unwrap()
        .output(input.0, input.1, input.2, input.3, TensorFormat::NCHW)
        .unwrap()
        .scale_bias_mean_var(input.1, TensorFormat::NCHW)
        .unwrap()
        .build();
    batch_norm
}

#[cfg(feature = "nvidia")]
fn create_batch_norm_backward_gpu<T: Num>(input: DimDyn) -> BatchNorm2dBackward<T> {
    let input = (
        input[0].try_into().unwrap(),
        input[1].try_into().unwrap(),
        input[2].try_into().unwrap(),
        input[3].try_into().unwrap(),
    );
    let batch_norm_backward = BatchNorm2dBackwardBuilder::<T>::new()
        .input(input.0, input.1, input.2, input.3, TensorFormat::NCHW)
        .unwrap()
        .input_grad(input.0, input.1, input.2, input.3, TensorFormat::NCHW)
        .unwrap()
        .output_grad(input.0, input.1, input.2, input.3, TensorFormat::NCHW)
        .unwrap()
        .scale_bias_mean_var(input.1, TensorFormat::NCHW)
        .unwrap()
        .build();
    batch_norm_backward
}

#[cfg(feature = "nvidia")]
fn create_batch_norm_inference_gpu<T: Num>(input: DimDyn) -> BatchNorm2dInference<T> {
    let input = (
        input[0].try_into().unwrap(),
        input[1].try_into().unwrap(),
        input[2].try_into().unwrap(),
        input[3].try_into().unwrap(),
    );
    let batch_norm_inference = BatchNorm2dInferenceBuilder::<T>::new()
        .input(input.0, input.1, input.2, input.3, TensorFormat::NCHW)
        .unwrap()
        .output(input.0, input.1, input.2, input.3, TensorFormat::NCHW)
        .unwrap()
        .scale_bias_mean_var(input.1, TensorFormat::NCHW)
        .unwrap()
        .build();
    batch_norm_inference
}

pub trait BatchNormalization: DeviceBase {
    fn batch_norm_2d_forward_train<T: Num>(
        momentum: f64,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
        mean: Matrix<Ref<&mut T>, DimDyn, Self>,
        variance: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_mean: Option<Matrix<Ref<&mut T>, DimDyn, Self>>,
        saving_inv_variance: Option<Matrix<Ref<&mut T>, DimDyn, Self>>,
        device_batch_norm: Option<BatchNorm2dConfig<T>>,
    );

    fn batch_norm_2d_backward<T: Num>(
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y_grad: Matrix<Ref<&T>, DimDyn, Self>,
        x_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        scale_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        bias_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_mean: Option<Matrix<Ref<&T>, DimDyn, Self>>,
        saving_inv_variance: Option<Matrix<Ref<&T>, DimDyn, Self>>,
        device_batch_norm_backward: Option<BatchNorm2dBackwardConfig<T>>,
    );

    fn bach_norm_2d_forward_inference<T: Num>(
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
        mean: Matrix<Ref<&T>, DimDyn, Self>,
        variance: Matrix<Ref<&T>, DimDyn, Self>,
        device_batch_norm_inference: Option<BatchNorm2dInferenceConfig<T>>,
    );
}

#[cfg(feature = "nvidia")]
impl BatchNormalization for Nvidia {
    fn batch_norm_2d_forward_train<T: Num>(
        momentum: f64,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
        mean: Matrix<Ref<&mut T>, DimDyn, Self>,
        variance: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_mean: Option<Matrix<Ref<&mut T>, DimDyn, Self>>,
        saving_inv_variance: Option<Matrix<Ref<&mut T>, DimDyn, Self>>,
        device_batch_norm: Option<BatchNorm2dConfig<T>>,
    ) {
        let momentum = 1. - momentum;
        let batch_norm = match device_batch_norm {
            Some(ref batch_norm) => &batch_norm.device_batch_norm,
            None => &create_batch_norm_gpu::<T>(x.shape()),
        };
        let saving_mean = match saving_mean {
            Some(saved_mean) => saved_mean.as_mut_ptr(),
            None => std::ptr::null_mut(),
        };
        let saving_inv_variance = match saving_inv_variance {
            Some(saved_inv_variance) => saved_inv_variance.as_mut_ptr(),
            None => std::ptr::null_mut(),
        };
        batch_norm
            .forward_train(
                T::one(),
                T::zero(),
                x.as_ptr(),
                y.as_mut_ptr(),
                scale.as_ptr(),
                bias.as_ptr(),
                mean.as_mut_ptr(),
                variance.as_mut_ptr(),
                momentum,
                saving_mean,
                saving_inv_variance,
            )
            .unwrap();
    }

    fn batch_norm_2d_backward<T: Num>(
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y_grad: Matrix<Ref<&T>, DimDyn, Self>,
        x_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        scale_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        bias_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_mean: Option<Matrix<Ref<&T>, DimDyn, Self>>,
        saving_inv_variance: Option<Matrix<Ref<&T>, DimDyn, Self>>,
        device_batch_norm_backward: Option<BatchNorm2dBackwardConfig<T>>,
    ) {
        let batch_norm_backward = match device_batch_norm_backward {
            Some(ref batch_norm_backward) => &batch_norm_backward.device_batch_norm_backward,
            None => &create_batch_norm_backward_gpu::<T>(x.shape()),
        };
        let saving_mean = match saving_mean {
            Some(saved_mean) => saved_mean.as_ptr(),
            None => std::ptr::null_mut(),
        };
        let saving_inv_variance = match saving_inv_variance {
            Some(saved_inv_variance) => saved_inv_variance.as_ptr(),
            None => std::ptr::null_mut(),
        };
        batch_norm_backward
            .backward(
                T::one(),
                T::zero(),
                T::one(),
                T::zero(),
                x.as_ptr(),
                y_grad.as_ptr(),
                x_grad.as_mut_ptr(),
                scale.as_ptr(),
                scale_grad.as_mut_ptr(),
                bias_grad.as_mut_ptr(),
                saving_mean,
                saving_inv_variance,
            )
            .unwrap();
    }

    fn bach_norm_2d_forward_inference<T: Num>(
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
        mean: Matrix<Ref<&T>, DimDyn, Self>,
        variance: Matrix<Ref<&T>, DimDyn, Self>,
        device_batch_norm_inference: Option<BatchNorm2dInferenceConfig<T>>,
    ) {
        let batch_norm_inference = match device_batch_norm_inference {
            Some(ref batch_norm_inference) => &batch_norm_inference.device_batch_norm_inference,
            None => &create_batch_norm_inference_gpu::<T>(x.shape()),
        };
        batch_norm_inference
            .forward_inference(
                T::one(),
                T::zero(),
                x.as_ptr(),
                y.as_mut_ptr(),
                scale.as_ptr(),
                bias.as_ptr(),
                mean.as_ptr(),
                variance.as_ptr(),
            )
            .unwrap();
    }
}

impl BatchNormalization for Cpu {
    fn batch_norm_2d_forward_train<T: Num>(
        momentum: f64,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
        mean: Matrix<Ref<&mut T>, DimDyn, Self>,
        variance: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_mean: Option<Matrix<Ref<&mut T>, DimDyn, Self>>,
        saving_inv_variance: Option<Matrix<Ref<&mut T>, DimDyn, Self>>,
        _: Option<BatchNorm2dConfig<T>>,
    ) {
        let momentum = T::from_f64(momentum);
        let epsilon = T::from_f64(1e-10);
        let x_shape = x.shape();
        let n = x_shape[0] * x_shape[2] * x_shape[3];
        let c = x_shape[1];
        let x_transposed = x.transpose_by_index_new_matrix(&[0, 2, 3, 1]);
        let x_reshaped = x_transposed.reshape(&[n, c]);

        let num_elements = T::from_usize(x_reshaped.shape()[0]);

        let x_mean = x_reshaped.mean(Some(0), false);
        let x_diff = &x_reshaped - &x_mean;
        let x_variance = x_reshaped.variance(Some(0), false);
        let x_variance_unbiased = &x_variance * (num_elements / (num_elements - T::one()));

        let mean_t = &x_mean * (T::one() - momentum) + &mean * momentum;
        let variance_t = &x_variance_unbiased * (T::one() - momentum) + &variance * momentum;

        let inv_var = Matrix::<_, DimDyn, _>::ones(variance_t.shape()) / (&x_variance + epsilon);
        let inv_std = inv_var.sqrt();

        mean.copy_from(&mean_t);
        variance.copy_from(&variance_t);

        if let Some(saving_mean_mat) = saving_mean {
            saving_mean_mat.copy_from(&x_mean);
        }
        if let Some(saving_inv_variance_mat) = saving_inv_variance {
            saving_inv_variance_mat.copy_from(&inv_std);
        }

        let x_normalized = &x_diff * &inv_std;
        let y_tmp = &x_normalized * &scale + &bias;
        let y_transposed = y_tmp.reshape(&[x_shape[0], x_shape[2], x_shape[3], x_shape[1]]);
        y.copy_from(&y_transposed.transpose_by_index_new_matrix(&[0, 3, 1, 2]));
    }

    fn batch_norm_2d_backward<T: Num>(
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y_grad: Matrix<Ref<&T>, DimDyn, Self>,
        x_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        scale_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        bias_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_mean: Option<Matrix<Ref<&T>, DimDyn, Self>>,
        saving_inv_variance: Option<Matrix<Ref<&T>, DimDyn, Self>>,
        _: Option<BatchNorm2dBackwardConfig<T>>,
    ) {
        let epsilon = T::from_f64(1e-10);
        let n = x.shape()[0] * x.shape()[2] * x.shape()[3];
        let c = x.shape()[1];
        let x_shape = x.shape();

        // Transpose and reshape x and y_grad for easier manipulation
        let x_transposed = x.transpose_by_index_new_matrix(&[0, 2, 3, 1]);
        let x_reshaped = x_transposed.reshape(&[n, c]);

        let y_grad_transposed = y_grad.transpose_by_index_new_matrix(&[0, 2, 3, 1]);
        let y_grad_reshaped = y_grad_transposed.reshape(&[n, c]);

        let mean = if let Some(ref mean_mat) = saving_mean {
            mean_mat.new_matrix()
        } else {
            x_reshaped.mean(Some(0), false)
        };

        let inv_std = if let Some(ref inv_variance_mat) = saving_inv_variance {
            inv_variance_mat.new_matrix()
        } else {
            let x_variance = x_reshaped.variance(Some(0), false);
            let inv_var =
                Matrix::<_, DimDyn, _>::ones(x_variance.shape()) / (&x_variance + epsilon);
            inv_var.sqrt()
        };

        let x_centered = &x_reshaped - &mean;
        let x_hat = &x_centered * &inv_std;

        bias_grad.copy_from(&y_grad_reshaped.to_ref().sum(0, false));
        scale_grad.copy_from(&(&x_hat * &y_grad_reshaped).to_ref().sum(0, false));

        // Compute the gradients
        let term1 = &inv_std * &y_grad_reshaped * scale;
        let mut term2 = term1.to_ref().sum(0, false) / T::from_usize(n);
        term2.add_axis(0);
        let mut term3 =
            &x_centered * (&term1 * &x_centered).to_ref().sum(0, false) / T::from_usize(n);
        term3.add_axis(0);
        let term3 = term3 * &inv_std * &inv_std;

        let x_grad_reshaped = term1 - term2 - term3;

        let x_grad_transposed =
            x_grad_reshaped.reshape(&[x_shape[0], x_shape[2], x_shape[3], x_shape[1]]);

        x_grad.copy_from(&x_grad_transposed.transpose_by_index_new_matrix(&[0, 3, 1, 2]));
    }

    fn bach_norm_2d_forward_inference<T: Num>(
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
        mean: Matrix<Ref<&T>, DimDyn, Self>,
        variance: Matrix<Ref<&T>, DimDyn, Self>,
        _: Option<BatchNorm2dInferenceConfig<T>>,
    ) {
        let epsilon = T::from_f64(1e-10);
        let n = x.shape()[0] * x.shape()[2] * x.shape()[3];
        let c = x.shape()[1];
        let x_shape = x.shape();

        // Transpose and reshape x and y_grad for easier manipulation
        let x_transposed = x.transpose_by_index_new_matrix(&[0, 2, 3, 1]);
        let x_reshaped = x_transposed.reshape(&[n, c]);

        let mean = mean.to_ref();
        let inv_std = Matrix::<_, DimDyn, _>::ones(variance.shape()) / (&variance + epsilon).sqrt();

        let x_centered = &x_reshaped - mean;
        let x_hat = &x_centered * &inv_std;

        let y_tmp = &x_hat * &scale + &bias;
        let y_transposed = y_tmp.reshape(&[x_shape[0], x_shape[2], x_shape[3], x_shape[1]]);
        y.copy_from(&y_transposed.transpose_by_index_new_matrix(&[0, 3, 1, 2]));
    }
}

fn batch_norm_2d_shape_check(
    x: DimDyn,
    y: DimDyn,
    scale: DimDyn,
    bias: DimDyn,
    mean: DimDyn,
    variance: DimDyn,
    saving_mean: Option<DimDyn>,
    saving_inv_variance: Option<DimDyn>,
) -> Result<(), String> {
    if scale.len() != 1 {
        return Err("scale must be a vector".to_string());
    }
    if bias.len() != 1 {
        return Err("bias must be a vector".to_string());
    }
    if mean.len() != 1 {
        return Err("mean must be a vector".to_string());
    }
    if variance.len() != 1 {
        return Err("variance must be a vector".to_string());
    }
    if let Some(saving_mean) = saving_mean {
        if saving_mean.len() != 1 {
            return Err("saving_mean must be a vector".to_string());
        }
    }
    if let Some(saving_inv_variance) = saving_inv_variance {
        if saving_inv_variance.len() != 1 {
            return Err("saving_inv_variance must be a vector".to_string());
        }
    }
    if x.len() != 4 {
        return Err("x and y must have the same number of elements".to_string());
    }
    if x != y {
        return Err("x and y must have the same shape".to_string());
    }
    if x[1] != scale[0] {
        return Err("x and scale must have the same number of channels".to_string());
    }
    if x[1] != bias[0] {
        return Err("x and bias must have the same number of channels".to_string());
    }
    if x[1] != mean[0] {
        return Err("x and mean must have the same number of channels".to_string());
    }
    if x[1] != variance[0] {
        return Err("x and variance must have the same number of channels".to_string());
    }
    if let Some(saving_mean) = saving_mean {
        if x[1] != saving_mean[0] {
            return Err("x and saving_mean must have the same number of channels".to_string());
        }
    }
    if let Some(saving_inv_variance) = saving_inv_variance {
        if x[1] != saving_inv_variance[0] {
            return Err(
                "x and saving_inv_variance must have the same number of channels".to_string(),
            );
        }
    }
    Ok(())
}

fn batch_norm_2d_backward_shape_check(
    x: DimDyn,
    y_grad: DimDyn,
    x_grad: DimDyn,
    scale: DimDyn,
    scale_grad: DimDyn,
    bias_grad: DimDyn,
    saving_mean: Option<DimDyn>,
    saving_inv_variance: Option<DimDyn>,
) -> Result<(), String> {
    if scale.len() != 1 {
        return Err("scale must be a vector".to_string());
    }
    if bias_grad.len() != 1 {
        return Err("bias_grad must be a vector".to_string());
    }
    if let Some(saving_mean) = saving_mean {
        if saving_mean.len() != 1 {
            return Err("saving_mean must be a vector".to_string());
        }
    }
    if let Some(saving_inv_variance) = saving_inv_variance {
        if saving_inv_variance.len() != 1 {
            return Err("saving_inv_variance must be a vector".to_string());
        }
    }
    if x.len() != 4 {
        return Err("x and y_grad must have the same number of elements".to_string());
    }
    if x != y_grad {
        return Err("x and y_grad must have the same shape".to_string());
    }
    if x != x_grad {
        return Err("x and x_grad must have the same shape".to_string());
    }
    if x[1] != scale[0] {
        return Err("x and scale must have the same number of channels".to_string());
    }
    if x[1] != scale_grad[0] {
        return Err("x and scale_grad must have the same number of channels".to_string());
    }
    if x[1] != bias_grad[0] {
        return Err("x and bias_grad must have the same number of channels".to_string());
    }
    if let Some(saving_mean) = saving_mean {
        if x[1] != saving_mean[0] {
            return Err("x and saving_mean must have the same number of channels".to_string());
        }
    }
    if let Some(saving_inv_variance) = saving_inv_variance {
        if x[1] != saving_inv_variance[0] {
            return Err(
                "x and saving_inv_variance must have the same number of channels".to_string(),
            );
        }
    }
    Ok(())
}

pub fn try_batch_norm_2d_forward_trian<T: Num, D: Device>(
    momentum: f64,
    x: Matrix<Ref<&T>, DimDyn, D>,
    y: Matrix<Ref<&mut T>, DimDyn, D>,
    scale: Matrix<Ref<&T>, DimDyn, D>,
    bias: Matrix<Ref<&T>, DimDyn, D>,
    mean: Matrix<Ref<&mut T>, DimDyn, D>,
    variance: Matrix<Ref<&mut T>, DimDyn, D>,
    saving_mean: Option<Matrix<Ref<&mut T>, DimDyn, D>>,
    saving_inv_variance: Option<Matrix<Ref<&mut T>, DimDyn, D>>,
    device_batch_norm: Option<BatchNorm2dConfig<T>>,
) -> Result<(), String> {
    let x_shape = x.shape();
    let y_shape = y.shape();
    let scale_shape = scale.shape();
    let bias_shape = bias.shape();
    let mean_shape = mean.shape();
    let variance_shape = variance.shape();
    let saving_mean_shape = saving_mean.as_ref().map(|x| x.shape());
    let saving_inv_variance_shape = saving_inv_variance.as_ref().map(|x| x.shape());

    if let Err(e) = batch_norm_2d_shape_check(
        x_shape,
        y_shape,
        scale_shape,
        bias_shape,
        mean_shape,
        variance_shape,
        saving_mean_shape,
        saving_inv_variance_shape,
    ) {
        return Err(e);
    }

    D::batch_norm_2d_forward_train(
        momentum,
        x,
        y,
        scale,
        bias,
        mean,
        variance,
        saving_mean,
        saving_inv_variance,
        device_batch_norm,
    );

    Ok(())
}

pub fn try_batch_norm_2d_forward_inference<T: Num, D: Device>(
    x: Matrix<Ref<&T>, DimDyn, D>,
    y: Matrix<Ref<&mut T>, DimDyn, D>,
    scale: Matrix<Ref<&T>, DimDyn, D>,
    bias: Matrix<Ref<&T>, DimDyn, D>,
    mean: Matrix<Ref<&T>, DimDyn, D>,
    variance: Matrix<Ref<&T>, DimDyn, D>,
    device_batch_norm_inference: Option<BatchNorm2dInferenceConfig<T>>,
) -> Result<(), String> {
    let x_shape = x.shape();
    let y_shape = y.shape();
    let scale_shape = scale.shape();
    let bias_shape = bias.shape();
    let mean_shape = mean.shape();
    let variance_shape = variance.shape();

    if let Err(e) = batch_norm_2d_shape_check(
        x_shape,
        y_shape,
        scale_shape,
        bias_shape,
        mean_shape,
        variance_shape,
        None,
        None,
    ) {
        return Err(e);
    }

    D::bach_norm_2d_forward_inference(
        x,
        y,
        scale,
        bias,
        mean,
        variance,
        device_batch_norm_inference,
    );

    Ok(())
}

pub fn try_batch_norm_2d_backward<T: Num, D: Device>(
    x: Matrix<Ref<&T>, DimDyn, D>,
    y_grad: Matrix<Ref<&T>, DimDyn, D>,
    x_grad: Matrix<Ref<&mut T>, DimDyn, D>,
    scale: Matrix<Ref<&T>, DimDyn, D>,
    scale_grad: Matrix<Ref<&mut T>, DimDyn, D>,
    bias_grad: Matrix<Ref<&mut T>, DimDyn, D>,
    saving_mean: Option<Matrix<Ref<&T>, DimDyn, D>>,
    saving_inv_variance: Option<Matrix<Ref<&T>, DimDyn, D>>,
    device_batch_norm_backward: Option<BatchNorm2dBackwardConfig<T>>,
) -> Result<(), String> {
    let x_shape = x.shape();
    let y_grad_shape = y_grad.shape();
    let x_grad_shape = x_grad.shape();
    let scale_shape = scale.shape();
    let scale_grad_shape = scale_grad.shape();
    let bias_grad_shape = bias_grad.shape();
    let saving_mean_shape = saving_mean.as_ref().map(|x| x.shape());
    let saving_inv_variance_shape = saving_inv_variance.as_ref().map(|x| x.shape());

    if let Err(e) = batch_norm_2d_backward_shape_check(
        x_shape,
        y_grad_shape,
        x_grad_shape,
        scale_shape,
        scale_grad_shape,
        bias_grad_shape,
        saving_mean_shape,
        saving_inv_variance_shape,
    ) {
        return Err(e);
    }

    D::batch_norm_2d_backward(
        x,
        y_grad,
        x_grad,
        scale,
        scale_grad,
        bias_grad,
        saving_mean,
        saving_inv_variance,
        device_batch_norm_backward,
    );

    Ok(())
}

pub fn batch_norm_2d_forward_train<T: Num, D: Device>(
    momentum: f64,
    x: Matrix<Ref<&T>, DimDyn, D>,
    y: Matrix<Ref<&mut T>, DimDyn, D>,
    scale: Matrix<Ref<&T>, DimDyn, D>,
    bias: Matrix<Ref<&T>, DimDyn, D>,
    mean: Matrix<Ref<&mut T>, DimDyn, D>,
    variance: Matrix<Ref<&mut T>, DimDyn, D>,
    saving_mean: Option<Matrix<Ref<&mut T>, DimDyn, D>>,
    saving_inv_variance: Option<Matrix<Ref<&mut T>, DimDyn, D>>,
    device_batch_norm: Option<BatchNorm2dConfig<T>>,
) {
    try_batch_norm_2d_forward_trian(
        momentum,
        x,
        y,
        scale,
        bias,
        mean,
        variance,
        saving_mean,
        saving_inv_variance,
        device_batch_norm,
    )
    .unwrap();
}

pub fn batch_norm_2d_forward_inference<T: Num, D: Device>(
    x: Matrix<Ref<&T>, DimDyn, D>,
    y: Matrix<Ref<&mut T>, DimDyn, D>,
    scale: Matrix<Ref<&T>, DimDyn, D>,
    bias: Matrix<Ref<&T>, DimDyn, D>,
    mean: Matrix<Ref<&T>, DimDyn, D>,
    variance: Matrix<Ref<&T>, DimDyn, D>,
    device_batch_norm_inference: Option<BatchNorm2dInferenceConfig<T>>,
) {
    try_batch_norm_2d_forward_inference(
        x,
        y,
        scale,
        bias,
        mean,
        variance,
        device_batch_norm_inference,
    )
    .unwrap();
}

pub fn batch_norm_2d_backward<T: Num, D: Device>(
    x: Matrix<Ref<&T>, DimDyn, D>,
    y_grad: Matrix<Ref<&T>, DimDyn, D>,
    x_grad: Matrix<Ref<&mut T>, DimDyn, D>,
    scale: Matrix<Ref<&T>, DimDyn, D>,
    scale_grad: Matrix<Ref<&mut T>, DimDyn, D>,
    bias_grad: Matrix<Ref<&mut T>, DimDyn, D>,
    saving_mean: Option<Matrix<Ref<&T>, DimDyn, D>>,
    saving_inv_variance: Option<Matrix<Ref<&T>, DimDyn, D>>,
    device_batch_norm_backward: Option<BatchNorm2dBackwardConfig<T>>,
) {
    try_batch_norm_2d_backward(
        x,
        y_grad,
        x_grad,
        scale,
        scale_grad,
        bias_grad,
        saving_mean,
        saving_inv_variance,
        device_batch_norm_backward,
    )
    .unwrap();
}

#[cfg(test)]
mod batch_norm {
    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    use zenu_test::*;

    use super::*;

    #[derive(Debug)]
    struct BatchNormInputs<D: Device> {
        x: Matrix<Owned<f32>, DimDyn, D>,
        y: Matrix<Owned<f32>, DimDyn, D>,
        scale: Matrix<Owned<f32>, DimDyn, D>,
        bias: Matrix<Owned<f32>, DimDyn, D>,
        mean: Matrix<Owned<f32>, DimDyn, D>,
        variance: Matrix<Owned<f32>, DimDyn, D>,
        saved_mean: Matrix<Owned<f32>, DimDyn, D>,
        saved_variance: Matrix<Owned<f32>, DimDyn, D>,
    }

    fn small_data<D: Device>() -> BatchNormInputs<D> {
        let x = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![
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
            ],
            &[2, 2, 2, 2],
        );
        let y = vec![
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
        let running_mean = vec![-0.36513, 0.15035464];
        let running_variance = vec![0.4431935, 1.0805689];
        let saved_mean = vec![-0.40570003, 0.16706072];
        let saved_variance = vec![1.5234232, 0.97564316];
        let scale = vec![1.0, 1.0];
        let bias = vec![0.0, 0.0];
        let y = Matrix::<Owned<f32>, DimDyn, D>::from_vec(y, &[2, 2, 2, 2]);
        let mean = Matrix::<Owned<f32>, DimDyn, D>::from_vec(running_mean, &[2]);
        let variance = Matrix::<Owned<f32>, DimDyn, D>::from_vec(running_variance, &[2]);
        let scale = Matrix::<Owned<f32>, DimDyn, D>::from_vec(scale, &[2]);
        let bias = Matrix::<Owned<f32>, DimDyn, D>::from_vec(bias, &[2]);
        let saved_mean = Matrix::<Owned<f32>, DimDyn, D>::from_vec(saved_mean, &[2]);
        let saved_variance = Matrix::<Owned<f32>, DimDyn, D>::from_vec(saved_variance, &[2]);
        BatchNormInputs {
            x,
            y,
            scale,
            bias,
            mean,
            variance,
            saved_mean,
            saved_variance,
        }
    }

    fn small_foward<D: Device>() {
        let inputs = small_data::<D>();
        let mut y_out = Matrix::<Owned<f32>, DimDyn, D>::zeros(inputs.y.shape());
        let mut mean_out = Matrix::<Owned<f32>, DimDyn, D>::zeros(inputs.mean.shape());
        let mut variance_out = Matrix::<Owned<f32>, DimDyn, D>::zeros(inputs.variance.shape());
        let mut saved_mean_out = Matrix::<Owned<f32>, DimDyn, D>::zeros(inputs.saved_mean.shape());
        let mut saved_variance_out =
            Matrix::<Owned<f32>, DimDyn, D>::zeros(inputs.saved_variance.shape());
        let batch_norm = BatchNorm2dConfig::<f32>::new(inputs.x.shape());
        D::batch_norm_2d_forward_train(
            0.1,
            inputs.x.to_ref(),
            y_out.to_ref_mut(),
            inputs.scale.to_ref(),
            inputs.bias.to_ref(),
            mean_out.to_ref_mut(),
            variance_out.to_ref_mut(),
            Some(saved_mean_out.to_ref_mut()),
            Some(saved_variance_out.to_ref_mut()),
            Some(batch_norm),
        );

        assert_mat_eq_epsilon!(y_out.to_ref(), inputs.y.to_ref(), 2e-4);
        assert_mat_eq_epsilon!(mean_out.to_ref(), inputs.mean.to_ref(), 2e-4);
        assert_mat_eq_epsilon!(variance_out.to_ref(), inputs.variance.to_ref(), 2e-4);
        assert_mat_eq_epsilon!(saved_mean_out.to_ref(), inputs.saved_mean.to_ref(), 2e-4);
        assert_mat_eq_epsilon!(
            saved_variance_out.to_ref(),
            inputs.saved_variance.to_ref(),
            2e-4
        );
    }
    run_mat_test!(small_foward, small_forward_cpu, small_forward_gpu);

    #[derive(Debug)]
    struct BatchNormBackward<D: Device> {
        x: Matrix<Owned<f32>, DimDyn, D>,
        y_grad: Matrix<Owned<f32>, DimDyn, D>,
        scale: Matrix<Owned<f32>, DimDyn, D>,
        saved_mean: Matrix<Owned<f32>, DimDyn, D>,
        saved_variance: Matrix<Owned<f32>, DimDyn, D>,
    }

    fn small_data_backward<D: Device>() -> BatchNormBackward<D> {
        let x = vec![
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
        let y_grad = vec![
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
        let saved_mean = vec![-0.04057, 0.01670607];
        let saved_variance = vec![0.9492437, 1.0200632];
        let scale = vec![1.0, 1.0];
        let x = Matrix::<Owned<f32>, DimDyn, D>::from_vec(x, &[2, 2, 2, 2]);
        let y_grad = Matrix::<Owned<f32>, DimDyn, D>::from_vec(y_grad, &[2, 2, 2, 2]);
        let scale = Matrix::<Owned<f32>, DimDyn, D>::from_vec(scale, &[2]);
        let saved_mean = Matrix::<Owned<f32>, DimDyn, D>::from_vec(saved_mean, &[2]);
        let saved_variance = Matrix::<Owned<f32>, DimDyn, D>::from_vec(saved_variance, &[2]);
        BatchNormBackward {
            x,
            y_grad,
            scale,
            saved_mean,
            saved_variance,
        }
    }

    fn small_backward<D: Device>() {
        let inputs = small_data_backward::<D>();
        let mut x_grad = Matrix::<Owned<f32>, DimDyn, D>::zeros(inputs.x.shape());
        let mut scale_grad = Matrix::<Owned<f32>, DimDyn, D>::zeros(inputs.scale.shape());
        let mut bias_grad = Matrix::<Owned<f32>, DimDyn, D>::zeros(inputs.scale.shape());
        let batch_norm_backward = BatchNorm2dBackwardConfig::<f32>::new(inputs.x.shape());
        D::batch_norm_2d_backward(
            inputs.x.to_ref(),
            inputs.y_grad.to_ref(),
            x_grad.to_ref_mut(),
            inputs.scale.to_ref(),
            scale_grad.to_ref_mut(),
            bias_grad.to_ref_mut(),
            Some(inputs.saved_mean.to_ref()),
            Some(inputs.saved_variance.to_ref()),
            Some(batch_norm_backward),
        );

        let x_grad_ans = vec![
            -0.06967929,
            0.41043705,
            -1.9042997,
            0.7856185,
            -0.39005604,
            -0.83055514,
            -0.69721717,
            -0.080333665,
            -0.54731166,
            0.4951802,
            -1.1199604,
            2.6264815,
            0.1793941,
            -0.52307177,
            0.99853456,
            1.131705,
        ];
        let scale_grad_ans = vec![2.0560942, 1.352522];
        let bias_grad_ans = vec![-4.6919003, 0.9442612];
        let x_grad_ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(x_grad_ans, &[2, 2, 2, 2]);
        let scale_grad_ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(scale_grad_ans, &[2]);
        let bias_grad_ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(bias_grad_ans, &[2]);
        assert_mat_eq_epsilon!(x_grad.to_ref(), x_grad_ans.to_ref(), 2e-4);
        assert_mat_eq_epsilon!(scale_grad.to_ref(), scale_grad_ans.to_ref(), 2e-4);
        assert_mat_eq_epsilon!(bias_grad.to_ref(), bias_grad_ans.to_ref(), 2e-4);
    }
    run_mat_test!(small_backward, small_backward_cpu, small_backward_gpu);

    fn small_foward_inference<D: Device>() {
        let inputs = small_forward_inference_data::<f32, D>();
        let mut y_out = Matrix::<Owned<f32>, DimDyn, D>::zeros(inputs.y.shape());
        let batch_norm_inference = BatchNorm2dInferenceConfig::<f32>::new(inputs.x.shape());
        D::bach_norm_2d_forward_inference(
            inputs.x.to_ref(),
            y_out.to_ref_mut(),
            inputs.scale.to_ref(),
            inputs.bias.to_ref(),
            inputs.mean.to_ref(),
            inputs.variance.to_ref(),
            Some(batch_norm_inference),
        );

        assert_mat_eq_epsilon!(y_out.to_ref(), inputs.y.to_ref(), 3e-3);
    }
    run_mat_test!(
        small_foward_inference,
        small_forward_inference_cpu,
        small_forward_inference_gpu
    );

    #[derive(Debug)]
    struct ForwardInputs<T: Num, D: Device> {
        x: Matrix<Owned<T>, DimDyn, D>,
        y: Matrix<Owned<T>, DimDyn, D>,
        scale: Matrix<Owned<T>, DimDyn, D>,
        bias: Matrix<Owned<T>, DimDyn, D>,
        mean: Matrix<Owned<T>, DimDyn, D>,
        variance: Matrix<Owned<T>, DimDyn, D>,
    }

    fn small_forward_inference_data<T: Num, D: Device>() -> ForwardInputs<T, D> {
        let x = vec![
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
        let y = vec![
            -0.6203, -0.5908, -1.5910, -1.3877, 3.3524, 2.9482, 0.3480, -4.2931, -2.2263, -0.4678,
            -2.2570, -2.2106, 1.4723, 4.3557, 4.0439, 0.5253,
        ];
        let mean = vec![-0.7193, -0.4033];
        let variance = vec![0.5966, 0.1820];
        let scale = vec![-0.8567, 1.1006];
        let bias = vec![-1.0712, 0.1227];

        let x = x.into_iter().map(T::from_f64).collect();
        let y = y.into_iter().map(T::from_f64).collect();
        let mean = mean.into_iter().map(T::from_f64).collect();
        let variance = variance.into_iter().map(T::from_f64).collect();
        let scale = scale.into_iter().map(T::from_f64).collect();
        let bias = bias.into_iter().map(T::from_f64).collect();

        let x = Matrix::<Owned<T>, DimDyn, D>::from_vec(x, &[2, 2, 2, 2]);
        let y = Matrix::<Owned<T>, DimDyn, D>::from_vec(y, &[2, 2, 2, 2]);
        let mean = Matrix::<Owned<T>, DimDyn, D>::from_vec(mean, &[2]);
        let variance = Matrix::<Owned<T>, DimDyn, D>::from_vec(variance, &[2]);
        let scale = Matrix::<Owned<T>, DimDyn, D>::from_vec(scale, &[2]);
        let bias = Matrix::<Owned<T>, DimDyn, D>::from_vec(bias, &[2]);
        ForwardInputs {
            x,
            y,
            scale,
            bias,
            mean,
            variance,
        }
    }
}

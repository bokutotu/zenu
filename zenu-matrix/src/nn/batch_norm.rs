use crate::{
    device::{cpu::Cpu, Device},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Ref},
    num::Num,
};

#[cfg(feature = "nvidia")]
use zenu_cuda::cudnn::{batch_norm::*, TensorFormat};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

#[cfg(feature = "nvidia")]
fn batch_norm2d_forward_train_gpu<T: Num>(
    momentum: f64,
    x: Matrix<Ref<&T>, DimDyn, Nvidia>,
    y: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    scale: Matrix<Ref<&T>, DimDyn, Nvidia>,
    bias: Matrix<Ref<&T>, DimDyn, Nvidia>,
    mean: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    variance: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    saving_mean: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    saving_inv_variance: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    batch_norm: Option<BatchNorm2d<T>>,
) {
    let momentum = 1. - momentum;
    let alpha = T::one() - T::from_f64(momentum);
    let beta = T::from_f64(momentum);
    match batch_norm {
        Some(batch_norm) => batch_norm
            .forward_train(
                // alpha,
                // beta,
                T::one(),
                T::zero(),
                x.as_ptr(),
                y.as_mut_ptr(),
                scale.as_ptr(),
                bias.as_ptr(),
                mean.as_mut_ptr(),
                variance.as_mut_ptr(),
                momentum,
                saving_mean.as_mut_ptr(),
                saving_inv_variance.as_mut_ptr(),
            )
            .unwrap(),
        None => panic!("batch_norm is None"),
    };
}

#[cfg(feature = "nvidia")]
fn batch_norm2d_backward_gpu<T: Num>(
    momentum: T,
    x: Matrix<Ref<&T>, DimDyn, Nvidia>,
    y_grad: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    x_grad: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    scale: Matrix<Ref<&T>, DimDyn, Nvidia>,
    scale_grad: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    bias_grad: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    saving_mean: Matrix<Ref<&T>, DimDyn, Nvidia>,
    saving_inv_variance: Matrix<Ref<&T>, DimDyn, Nvidia>,
    batch_norm_backward: Option<BatchNorm2dBackward<T>>,
) {
    let alpha = T::one() - momentum;
    let beta = momentum;
    match batch_norm_backward {
        Some(batch_norm_backward) => batch_norm_backward
            .backward(
                alpha,
                beta,
                alpha,
                beta,
                x.as_ptr(),
                y_grad.as_mut_ptr(),
                x_grad.as_mut_ptr(),
                scale.as_ptr(),
                scale_grad.as_mut_ptr(),
                bias_grad.as_mut_ptr(),
                saving_mean.as_ptr(),
                saving_inv_variance.as_ptr(),
            )
            .unwrap(),
        None => panic!("batch_norm_backward is None"),
    };
}

#[cfg(feature = "nvidia")]
fn create_batch_norm_gpu<T: Num>(input: (usize, usize, usize, usize)) -> BatchNorm2d<T> {
    let input = (
        input.0.try_into().unwrap(),
        input.1.try_into().unwrap(),
        input.2.try_into().unwrap(),
        input.3.try_into().unwrap(),
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
fn create_batch_norm_backward_gpu<T: Num>(
    input: (usize, usize, usize, usize),
) -> BatchNorm2dBackward<T> {
    let input = (
        input.0.try_into().unwrap(),
        input.1.try_into().unwrap(),
        input.2.try_into().unwrap(),
        input.3.try_into().unwrap(),
    );
    let batch_norm_backward = BatchNorm2dBackwardBuilder::<T>::new()
        .input(input.0, input.1, input.2, input.3, TensorFormat::NCHW)
        .unwrap()
        .output_grad(input.0, input.1, input.2, input.3, TensorFormat::NCHW)
        .unwrap()
        .scale_bias_mean_var(input.1, TensorFormat::NCHW)
        .unwrap()
        .build();
    batch_norm_backward
}

fn batch_norm2d_forward_train_cpu<T: Num>(
    momentum: T,
    x: Matrix<Ref<&T>, DimDyn, Cpu>,
    y: Matrix<Ref<&mut T>, DimDyn, Cpu>,
    scale: Matrix<Ref<&T>, DimDyn, Cpu>,
    bias: Matrix<Ref<&T>, DimDyn, Cpu>,
    mean: Matrix<Ref<&mut T>, DimDyn, Cpu>,
    variance: Matrix<Ref<&mut T>, DimDyn, Cpu>,
    saving_mean: Matrix<Ref<&mut T>, DimDyn, Cpu>,
    saving_inv_variance: Matrix<Ref<&mut T>, DimDyn, Cpu>,
) {
    let epsilon = T::from_f64(1e-10);
    let x_transposed = x.transpose_by_index_new_matrix(&[0, 2, 3, 1]);
    let x_reshaped = x_transposed.reshape(&[
        x_transposed.shape()[0] * x_transposed.shape()[2] * x_transposed.shape()[3],
        x_transposed.shape()[1],
    ]);

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

    saving_mean.copy_from(&x_mean);
    saving_inv_variance.copy_from(&inv_std);

    let x_normalized = &x_diff * &inv_std;
    let y_tmp = &x_normalized * &scale + &bias;
    let y_transposed = y_tmp.reshape(&[
        x_transposed.shape()[0],
        x_transposed.shape()[2],
        x_transposed.shape()[3],
        x_transposed.shape()[1],
    ]);
    y.copy_from(&y_transposed.transpose_by_index_new_matrix(&[0, 3, 1, 2]));
}

fn batch_norm2d_backward_cpu<T: Num>(
    momentum: T,
    x: Matrix<Ref<&T>, DimDyn, Cpu>,
    x_grad: Matrix<Ref<&mut T>, DimDyn, Cpu>,
    y_grad: Matrix<Ref<&T>, DimDyn, Cpu>,
    scale: Matrix<Ref<&T>, DimDyn, Cpu>,
    scale_grad: Matrix<Ref<&mut T>, DimDyn, Cpu>,
    bias_grad: Matrix<Ref<&mut T>, DimDyn, Cpu>,
    epsilon: f64,
    saving_mean: Matrix<Ref<&T>, DimDyn, Cpu>,
    saving_inv_variance: Matrix<Ref<&T>, DimDyn, Cpu>,
) {
    let epsilon = T::from_f64(1e-10);
    let batch_size = T::from_usize(x.shape()[0]);

    let x_transposed = x.transpose_by_index_new_matrix(&[0, 2, 3, 1]);
    let x_reshaped = x_transposed.reshape(&[
        x_transposed.shape()[0] * x_transposed.shape()[2] * x_transposed.shape()[3],
        x_transposed.shape()[1],
    ]);

    let y_grad_transposed = y_grad.transpose_by_index_new_matrix(&[0, 2, 3, 1]);
    let y_grad_reshaped = y_grad_transposed.reshape(&[
        y_grad_transposed.shape()[0] * y_grad_transposed.shape()[2] * y_grad_transposed.shape()[3],
        y_grad_transposed.shape()[1],
    ]);

    let xc = (&x_reshaped - &saving_mean) * &saving_inv_variance;

    bias_grad.copy_from(&y_grad_transposed.to_ref().sum(0, false));
    scale_grad.copy_from(&(&xc * &y_grad_reshaped).to_ref().sum(0, false));

    let tmp_x_grad = &y_grad_reshaped / batch_size - &xc * &scale_grad / batch_size;
    let tmp_x_grad = &tmp_x_grad * &saving_inv_variance;

    let x_grad_transposed = tmp_x_grad.reshape(&[
        x_transposed.shape()[0],
        x_transposed.shape()[2],
        x_transposed.shape()[3],
        x_transposed.shape()[1],
    ]);
    x_grad.copy_from(&x_grad_transposed.transpose_by_index_new_matrix(&[0, 3, 1, 2]));
}

pub trait BatchNormalization: Device {
    fn forward_train<T: Num, B>(
        momentum: T,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
        mean: Matrix<Ref<&mut T>, DimDyn, Self>,
        variance: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_mean: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_inv_variance: Matrix<Ref<&mut T>, DimDyn, Self>,
        device_batch_norm: Option<B>,
    );

    fn backward<T: Num, B>(
        momentum: T,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        x_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        scale_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        bias_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_mean: Matrix<Ref<&T>, DimDyn, Self>,
        saving_inv_variance: Matrix<Ref<&T>, DimDyn, Self>,
        device_batch_norm_backward: Option<B>,
    );
}

impl BatchNormalization for Cpu {
    fn forward_train<T: Num, B>(
        momentum: T,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
        mean: Matrix<Ref<&mut T>, DimDyn, Self>,
        variance: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_mean: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_inv_variance: Matrix<Ref<&mut T>, DimDyn, Self>,
        _: Option<B>,
    ) {
        batch_norm2d_forward_train_cpu(
            momentum,
            x,
            y,
            scale,
            bias,
            mean,
            variance,
            saving_mean,
            saving_inv_variance,
        );
    }

    fn backward<T: Num, B>(
        momentum: T,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        x_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        scale: Matrix<Ref<&T>, DimDyn, Self>,
        scale_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        bias_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        saving_mean: Matrix<Ref<&T>, DimDyn, Self>,
        saving_inv_variance: Matrix<Ref<&T>, DimDyn, Self>,
        _: Option<B>,
    ) {
        todo!();
    }
}

#[cfg(test)]
mod batch_norm {
    use crate::{
        device::{cpu::Cpu, Device},
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    use super::*;

    #[cfg(feature = "nvidia")]
    use zenu_cuda::cudnn::{batch_norm::*, TensorFormat};

    #[cfg(feature = "nvidia")]
    use crate::device::nvidia::Nvidia;

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
                // 0., 1., 2., 3., 4., 5., 6., 7., 0., 1., 2., 3., 4., 5., 6., 7.,
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
        let running_mean = vec![-0.04057, 0.01670607];
        let running_variance = vec![0.9492437, 1.0200632];
        let saved_mean = vec![-0.04057, 0.01670607];
        let saved_variance = vec![0.9492437, 1.0200632];
        let scale = vec![1.0, 1.0];
        let bias = vec![0.0, 0.0];
        let y = Matrix::<Owned<f32>, DimDyn, D>::zeros(&[2, 2, 2, 2]);
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

    #[test]
    fn small_cpu() {
        let mut inputs = small_data::<Cpu>();
        batch_norm2d_forward_train_cpu(
            0.5,
            inputs.x.to_ref(),
            inputs.y.to_ref_mut(),
            inputs.scale.to_ref(),
            inputs.bias.to_ref(),
            inputs.mean.to_ref_mut(),
            inputs.variance.to_ref_mut(),
            inputs.saved_mean.to_ref_mut(),
            inputs.saved_variance.to_ref_mut(),
        );

        println!("y {:?}", inputs.y);
        println!("mean {:?}", inputs.mean);
        println!("variance {:?}", inputs.variance);
        println!("saved mean {:?}", inputs.saved_mean);
        println!("saved variance {:?}", inputs.saved_variance);
        // panic!();
    }

    #[cfg(feature = "nvidia")]
    #[test]
    fn small_gpu() {
        let mut inputs = small_data::<Nvidia>();
        let batch_norm = BatchNorm2dBuilder::<f32>::new()
            .input(2, 2, 2, 2, TensorFormat::NCHW)
            .unwrap()
            .output(2, 2, 2, 2, TensorFormat::NCHW)
            .unwrap()
            .scale_bias_mean_var(2, TensorFormat::NCHW)
            .unwrap()
            .build();

        batch_norm2d_forward_train_gpu(
            0.5,
            inputs.x.to_ref(),
            inputs.y.to_ref_mut(),
            inputs.scale.to_ref(),
            inputs.bias.to_ref(),
            inputs.mean.to_ref_mut(),
            inputs.variance.to_ref_mut(),
            inputs.saved_mean.to_ref_mut(),
            inputs.saved_variance.to_ref_mut(),
            Some(batch_norm),
        );

        println!("y {:?}", inputs.y);
        println!("mean {:?}", inputs.mean);
        println!("variance {:?}", inputs.variance);
        println!("saved mean {:?}", inputs.saved_mean);
        println!("saved variance {:?}", inputs.saved_variance);
        // panic!();
    }
}

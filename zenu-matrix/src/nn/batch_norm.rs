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
    momentum: T,
    x: Matrix<Ref<&T>, DimDyn, Nvidia>,
    y: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    scale: Matrix<Ref<&T>, DimDyn, Nvidia>,
    bias: Matrix<Ref<&T>, DimDyn, Nvidia>,
    mean: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    variance: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    saving_mean: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    saving_inv_variance: Matrix<Ref<&mut T>, DimDyn, Nvidia>,
    epsilon: f64,
    batch_norm: Option<BatchNorm2d<T>>,
) {
    let alpha = T::one() - momentum;
    let beta = momentum;
    match batch_norm {
        Some(batch_norm) => batch_norm
            .forward_train(
                alpha,
                beta,
                x.as_ptr(),
                y.as_mut_ptr(),
                scale.as_ptr(),
                bias.as_ptr(),
                mean.as_mut_ptr(),
                variance.as_mut_ptr(),
                epsilon,
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
        .output(input.0, input.1, input.2, input.3, TensorFormat::NCHW)
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
    epsilon: f64,
    saving_mean: Matrix<Ref<&mut T>, DimDyn, Cpu>,
    saving_inv_variance: Matrix<Ref<&mut T>, DimDyn, Cpu>,
) {
    let epsilon = T::from_f64(epsilon);
    let x_transposed = x.transpose_by_index_new_matrix(&[0, 2, 3, 1]);
    let x_reshaped = x_transposed.reshape(&[
        x_transposed.shape()[0] * x_transposed.shape()[2] * x_transposed.shape()[3],
        x_transposed.shape()[1],
    ]);

    let num_elements = T::from_usize(x_reshaped.shape()[0]); // 行数を取得

    let x_mean = x_reshaped.mean(Some(0), false);
    let x_diff = &x_reshaped - &x_mean;
    let x_diff_squared = &x_diff * &x_diff;
    let x_variance = x_diff_squared.mean(Some(0), false) * num_elements / (num_elements - T::one());

    let inv_std = Matrix::<_, DimDyn, _>::ones(x_variance.shape()) / (x_variance.sqrt() + epsilon);
    let x_hat = &x_diff * &inv_std;
    let y_hat = x_hat * scale + bias;
    let y_reshaped = y_hat.reshape(&[x.shape()[0], x.shape()[2], x.shape()[3], x.shape()[1]]);
    let y_transposed = y_reshaped.transpose_by_index_new_matrix(&[0, 3, 1, 2]);
    y.copy_from(&y_transposed);

    let mean_t = &x_mean * (T::one() - momentum) + &mean * momentum;
    let variance_t = x_variance * (T::one() - momentum) + &variance * momentum;

    mean.copy_from(&mean_t);
    variance.copy_from(&variance_t);

    saving_mean.copy_from(&x_mean);
    saving_inv_variance.copy_from(&inv_std);
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
        epsilon: f64,
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
        epsilon: f64,
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
            epsilon,
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

    struct BatchNormInputs<D: Device> {
        x: Matrix<Owned<f32>, DimDyn, D>,
        y: Matrix<Owned<f32>, DimDyn, D>,
        scale: Matrix<Owned<f32>, DimDyn, D>,
        bias: Matrix<Owned<f32>, DimDyn, D>,
        mean: Matrix<Owned<f32>, DimDyn, D>,
        variance: Matrix<Owned<f32>, DimDyn, D>,
    }

    fn small_data<D: Device>() -> BatchNormInputs<D> {
        let x = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![
                0., 1., 2., 3., 4., 5., 6., 7., 0., 1., 2., 3., 4., 5., 6., 7.,
            ],
            &[2, 2, 2, 2],
        );
        let y = Matrix::<Owned<f32>, DimDyn, D>::zeros(x.shape());
        let scale = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 1.], [2]);
        let bias = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![0., 0.], [2]);
        let mean = Matrix::<Owned<f32>, DimDyn, D>::zeros([2]);
        let variance = Matrix::<Owned<f32>, DimDyn, D>::zeros([2]);
        BatchNormInputs {
            x,
            y,
            scale,
            bias,
            mean,
            variance,
        }
    }

    #[test]
    fn small_cpu() {
        let mut inputs = small_data::<Cpu>();
        let mut savig_mean = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[2]);
        let mut saving_inv_variance = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[2]);
        batch_norm2d_forward_train_cpu(
            0.0,
            inputs.x.to_ref(),
            inputs.y.to_ref_mut(),
            inputs.scale.to_ref(),
            inputs.bias.to_ref(),
            inputs.mean.to_ref_mut(),
            inputs.variance.to_ref_mut(),
            1e-5,
            savig_mean.to_ref_mut(),
            saving_inv_variance.to_ref_mut(),
        );
    }

    #[cfg(feature = "nvidia")]
    #[test]
    fn small_gpu() {
        let mut inputs = small_data::<Nvidia>();
        let mut savig_mean = Matrix::<Owned<f32>, DimDyn, Nvidia>::zeros(&[2]);
        let mut saving_inv_variance = Matrix::<Owned<f32>, DimDyn, Nvidia>::zeros(&[2]);
        let batch_norm = BatchNorm2dBuilder::<f32>::new()
            .input(2, 2, 2, 2, TensorFormat::NCHW)
            .unwrap()
            .output(2, 2, 2, 2, TensorFormat::NCHW)
            .unwrap()
            .scale_bias_mean_var(2, TensorFormat::NCHW)
            .unwrap()
            .build();

        batch_norm2d_forward_train_gpu(
            0.0,
            inputs.x.to_ref(),
            inputs.y.to_ref_mut(),
            inputs.scale.to_ref(),
            inputs.bias.to_ref(),
            inputs.mean.to_ref_mut(),
            inputs.variance.to_ref_mut(),
            savig_mean.to_ref_mut(),
            saving_inv_variance.to_ref_mut(),
            1.,
            Some(batch_norm),
        );
    }
}

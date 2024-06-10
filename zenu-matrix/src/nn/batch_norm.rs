use crate::{
    device::cpu::Cpu,
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Ref},
    num::Num,
};

#[cfg(feature = "nvidia")]
use zenu_cuda::cudnn::batch_norm::*;

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
                saving_variance.as_ptr(),
            )
            .unwrap(),
        None => panic!("batch_norm_backward is None"),
    };
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
    mut saving_mean: Matrix<Ref<&mut T>, DimDyn, Cpu>,
    mut saving_inv_variance: Matrix<Ref<&mut T>, DimDyn, Cpu>,
) {
    let x_transposed = x.transpose_by_index_new_matrix(&[0, 2, 3, 1]);
    let x_transposed = x_transposed.reshape(&[
        x_transposed.shape()[0],
        x_transposed.shape()[1] * x_transposed.shape()[2] * x_transposed.shape()[3],
    ]);
    let x_mean = x_transposed.mean(Some(0), false);
    let x_variance = x_transposed.variance(Some(0), false);
    let inv_std = Matrix::<_, DimDyn, _>::ones(x_variance.shape()) / x_variance.sqrt();
    let x_hat = (x_transposed - &x_mean) * &inv_std;
    let y_hat = x_hat * scale + bias;
    let y_hat = y_hat.reshape(&[y_hat.shape()[0], x.shape()[2], x.shape()[3], x.shape()[0]]);
    let y_hat = y_hat.transpose_by_index_new_matrix(&[3, 1, 2, 0]);
    y.copy_from(&y_hat);
    let mean_t = &x_mean * (T::one() - momentum) + &mean * momentum;
    let variance_t = x_variance * (T::one() - momentum) + &variance * momentum;
    mean.copy_from(&mean_t);
    variance.copy_from(&variance_t);
    saving_mean.copy_from(&x_mean);
    saving_inv_variance.copy_from(&inv_std);
}

#[cfg(test)]
mod batch_norm {
    use crate::{
        device::{cpu::Cpu, Device},
        dim::DimDyn,
        matrix::{Matrix, Owned},
        num::Num,
    };

    use super::batch_norm2d_forward_train_cpu;

    #[cfg(feature = "nvidia")]
    use zenu_cuda::cudnn::{batch_norm::*, TensorFormat};

    #[cfg(feature = "nvidia")]
    use crate::device::nvidia::Nvidia;

    struct BatchNormInputs<T: Num, D: Device> {
        x: Matrix<Owned<T>, DimDyn, D>,
        y: Matrix<Owned<T>, DimDyn, D>,
        scale: Matrix<Owned<T>, DimDyn, D>,
        bias: Matrix<Owned<T>, DimDyn, D>,
        mean: Matrix<Owned<T>, DimDyn, D>,
        variance: Matrix<Owned<T>, DimDyn, D>,
    }

    fn small_data<T: Num, D: Device>() -> BatchNormInputs<T, D> {
        let x = Matrix::<Owned<T>, DimDyn, D>::from_vec(
            vec![0., 1., 2., 3., 4., 5., 6., 7.],
            vec![1, 2, 2, 2],
        );
        let mut y = Matrix::<Owned<T>, DimDyn, D>::zeros(x.shape());
        let scale = Matrix::<Owned<T>, DimDyn, D>::from_vec(vec![1., 1.], vec![1, 2]);
        let bias = Matrix::<Owned<T>, DimDyn, D>::from_vec(vec![0., 0.], vec![1, 2]);
        let mut mean = Matrix::<Owned<T>, DimDyn, D>::zeros(vec![1, 2]);
        let mut variance = Matrix::<Owned<T>, DimDyn, D>::zeros(vec![1, 2]);
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
        let mut inputs = small_data::<f32, Cpu>();
        let mut savig_mean = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(vec![0., 0.]);
        let mut saving_inv_variance = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(vec![0., 0.]);
        batch_norm2d_forward_train_cpu(
            0.0,
            inputs.x.to_ref(),
            inputs.y.to_mut(),
            inputs.scale.to_ref(),
            inputs.bias.to_ref(),
            inputs.mean.to_ref_mut(),
            inputs.variance.to_ref_mut(),
            1e-5,
            savig_mean.to_ref_mut(),
            saving_inv_variance.to_ref_mut(),
        );

        println!("{:?}", inputs.y);
        println!("{:?}", inputs.mean);
        println!("{:?}", inputs.variance);
    }

    #[cfg(feature = "nvidia")]
    #[test]
    fn small_gpu() {
        let mut inputs = small_data::<f32, Nvidia>();
        let mut savig_mean = Matrix::<Owned<f32>, DimDyn, Nvidia>::zeros(vec![0., 0.]);
        let mut saving_inv_variance = Matrix::<Owned<f32>, DimDyn, Nvidia>::zeros(vec![0., 0.]);
        let batch_norm = BatchNorm2dBuilder::<f32>::new()
            .input(2, 2, 2, 2, TensorFormat::NCHW)
            .unwrap()
            .output(2, 2, 2, 2, TensorFormat::NCHW)
            .unwrap()
            .scale_bias(2, TensorFormat::NCHW)
            .unwrap()
            .build()
            .unwrap();

        batch_norm2d_forward_train_gpu(
            0.0,
            inputs.x.to_ref(),
            inputs.y.to_mut(),
            inputs.scale.to_ref(),
            inputs.bias.to_ref(),
            inputs.mean.to_ref_mut(),
            inputs.variance.to_ref_mut(),
            savig_mean.to_ref_mut(),
            saving_inv_variance.to_ref_mut(),
            1e-5,
            Some(batch_norm),
        );

        println!("{:?}", inputs.y);
        println!("{:?}", inputs.mean);
        println!("{:?}", inputs.variance);
    }
}

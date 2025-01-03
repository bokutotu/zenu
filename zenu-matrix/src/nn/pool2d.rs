use crate::{
    device::{cpu::Cpu, Device, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref},
    nn::col2im::col2im,
    num::Num,
};

#[cfg(feature = "nvidia")]
use zenu_cuda::cudnn::pooling::{Pool2d, PoolType};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

use super::im2col::{im2col, Im2ColRes};

pub struct Pool2dConfig<T: Num> {
    #[cfg(feature = "nvidia")]
    pub config: Pool2d<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Num> Pool2dConfig<T> {
    #[must_use]
    #[allow(unused_variables)]
    pub fn new(
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        input_shape: (usize, usize, usize, usize),
        output_shape: (usize, usize, usize, usize),
    ) -> Self {
        Self {
            #[cfg(feature = "nvidia")]
            config: Pool2d::<T>::new(
                PoolType::Max,
                kernel.0,
                kernel.1,
                padding.0,
                padding.1,
                stride.0,
                stride.1,
                input_shape,
                output_shape,
            )
            .unwrap(),
            _phantom: std::marker::PhantomData,
        }
    }
}

pub trait Pool2dImpl: DeviceBase {
    #[expect(clippy::missing_errors_doc)]
    fn pool2d<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        output: Matrix<Ref<&mut T>, DimDyn, Self>,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        config: &Pool2dConfig<T>,
    ) -> Result<(), String>;

    #[expect(clippy::too_many_arguments)]
    fn pool2d_backward<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        input_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        output: Matrix<Ref<&T>, DimDyn, Self>,
        output_grad: Matrix<Ref<&T>, DimDyn, Self>,
        kernel_shape: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        config: &Pool2dConfig<T>,
    );
}

impl Pool2dImpl for Cpu {
    fn pool2d<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        output: Matrix<Ref<&mut T>, DimDyn, Self>,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        _config: &Pool2dConfig<T>,
    ) -> Result<(), String> {
        let Im2ColRes { col, .. } = im2col(input, kernel, stride, padding, false);
        let col_shape = col.shape();
        let col = col.reshape_no_alloc_owned([
            col_shape[0],
            col_shape[1],
            col_shape[2] * col_shape[3],
            col_shape[4],
            col_shape[5],
        ]);

        output.copy_from(&col.max_axis(2, false));
        Ok(())
    }

    fn pool2d_backward<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        input_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        _output: Matrix<Ref<&T>, DimDyn, Self>,
        output_grad: Matrix<Ref<&T>, DimDyn, Self>,
        kernel_shape: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        _config: &Pool2dConfig<T>,
    ) {
        let gy_shape = output_grad.shape();
        let (kh, kw) = kernel_shape;
        let n = gy_shape[0];
        let c = gy_shape[1];
        let oh = gy_shape[2];
        let ow = gy_shape[3];

        let Im2ColRes { col, .. } = im2col(input, kernel_shape, stride, padding, false);
        let col_shape = col.shape();
        let mut gcol = Matrix::<Owned<T>, DimDyn, Self>::zeros(col.shape());
        let col = col.reshape_no_alloc_owned([
            col_shape[0],
            col_shape[1],
            col_shape[2] * col_shape[3],
            col_shape[4],
            col_shape[5],
        ]);
        let mut max_idxs = col.max_axis_idx_ravel(2);
        for (idx, max_idx) in max_idxs.iter_mut().enumerate() {
            *max_idx += kh * kw * idx;
        }

        for (idx, max_id) in max_idxs.iter().enumerate() {
            let grad_val = unsafe { output_grad.ptr().get_item(idx) };
            unsafe { gcol.to_ref_mut().ptr().assign_item(*max_id, grad_val) };
        }

        let gcol = gcol.reshape_no_alloc_owned([n, c, oh, ow, kh, kw]);
        let gcol = gcol.transpose_swap_index_new_matrix(2, 4);
        let gcol = gcol.transpose_swap_index_new_matrix(3, 5);

        let mut shape = [0; 4];
        for (i, itm) in input_grad.shape().slice().iter().enumerate() {
            shape[i] = *itm;
        }

        let col2im_tmp = col2im(gcol.to_ref(), shape, kernel_shape, stride, padding);
        input_grad.copy_from(&col2im_tmp);
    }
}

#[cfg(feature = "nvidia")]
impl Pool2dImpl for Nvidia {
    fn pool2d<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        output: Matrix<Ref<&mut T>, DimDyn, Self>,
        _kernel: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
        config: &Pool2dConfig<T>,
    ) -> Result<(), String> {
        config
            .config
            .forward(input.as_ptr(), output.as_mut_ptr(), T::one(), T::zero())
            .unwrap();
        Ok(())
    }

    fn pool2d_backward<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        input_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        output: Matrix<Ref<&T>, DimDyn, Self>,
        output_grad: Matrix<Ref<&T>, DimDyn, Self>,
        _kernel_shape: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
        config: &Pool2dConfig<T>,
    ) {
        config
            .config
            .backward(
                input.as_ptr(),
                input_grad.as_mut_ptr(),
                output.as_ptr(),
                output_grad.as_ptr(),
                T::one(),
                T::zero(),
            )
            .unwrap();
    }
}

#[must_use]
pub fn max_pool_2d_output_shape(
    input_shape: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> [usize; 4] {
    [
        input_shape[0],
        input_shape[1],
        (input_shape[2] + 2 * padding.0 - kernel.0) / stride.0 + 1,
        (input_shape[3] + 2 * padding.1 - kernel.1) / stride.1 + 1,
    ]
}

#[expect(clippy::missing_panics_doc, clippy::needless_pass_by_value)]
#[must_use]
pub fn max_pool_2d<T: Num, D: Device>(
    input: Matrix<Ref<&T>, DimDyn, D>,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    config: &Pool2dConfig<T>,
) -> Matrix<Owned<T>, DimDyn, D> {
    let output_shape = max_pool_2d_output_shape(input.shape().slice(), kernel, stride, padding);
    let mut output = Matrix::<Owned<T>, DimDyn, D>::zeros(output_shape);
    D::pool2d(
        input.to_ref(),
        output.to_ref_mut(),
        kernel,
        stride,
        padding,
        config,
    )
    .expect("pool2d failed");
    output
}

#[expect(clippy::needless_pass_by_value)]
#[must_use]
pub fn max_pool_2d_grad<T: Num, D: Device>(
    input: Matrix<Ref<&T>, DimDyn, D>,
    output: Matrix<Ref<&T>, DimDyn, D>,
    output_grad: Matrix<Ref<&T>, DimDyn, D>,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    config: &Pool2dConfig<T>,
) -> Matrix<Owned<T>, DimDyn, D> {
    let mut input_grad = Matrix::<Owned<T>, DimDyn, D>::zeros(input.shape());
    D::pool2d_backward(
        input.to_ref(),
        input_grad.to_ref_mut(),
        output.to_ref(),
        output_grad.to_ref(),
        kernel,
        stride,
        padding,
        config,
    );
    input_grad
}

#[expect(clippy::unreadable_literal, clippy::too_many_lines)]
#[cfg(test)]
mod pool2d {
    use zenu_test::{assert_mat_eq_epsilon, run_mat_test};

    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    use super::Pool2dConfig;

    fn device_forward<D: Device>() {
        let input = vec![
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
            2.3803675,
            -1.1256016,
            -0.3169981,
            -1.0924683,
            -0.0851943,
            -0.093348235,
            0.6870502,
            -0.83831537,
            0.018486667,
            -0.7504268,
            0.18540798,
            0.62113833,
            0.63818157,
            -0.24600095,
            2.3025165,
            -1.8816892,
        ];
        let output = vec![
            0.69200915, 1.2376579, 0.59883946, 1.8530061, 0.94629836, 1.5863342, 2.3022053,
            0.8728312, 1.0553575, 2.3803675, 0.6870502, 2.3025165,
        ];

        let input = Matrix::<Owned<f32>, DimDyn, D>::from_vec(input, [1, 3, 5, 5]);
        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(output, [1, 3, 2, 2]);

        let mut result = Matrix::<Owned<f32>, DimDyn, D>::zeros([1, 3, 2, 2]);

        let config = Pool2dConfig::new((3, 3), (2, 2), (0, 0), (1, 3, 5, 5), (1, 3, 2, 2));

        D::pool2d(
            input.to_ref(),
            result.to_ref_mut(),
            (3, 3),
            (2, 2),
            (0, 0),
            &config,
        )
        .unwrap();

        assert_mat_eq_epsilon!(result.clone(), ans, 1e-6);

        let output_grad = Matrix::<Owned<f32>, DimDyn, D>::ones_like(&result);

        let mut input_grad = Matrix::<Owned<f32>, DimDyn, D>::zeros_like(&input);

        D::pool2d_backward(
            input.to_ref(),
            input_grad.to_ref_mut(),
            result.to_ref(),
            output_grad.to_ref(),
            (3, 3),
            (2, 2),
            (0, 0),
            &config,
        );
        let input_grad_ans = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let input_grad_ans =
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(input_grad_ans, [1, 3, 5, 5]);
        assert_mat_eq_epsilon!(input_grad, input_grad_ans, 1e-6);
    }
    run_mat_test!(device_forward, device_forward_cpu, device_forward_nvidia);
}

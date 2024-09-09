use crate::{
    device::{cpu::Cpu, Device, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref},
    num::Num,
};

mod conv2d_bckwd_filter_cpu;
mod conv2d_cpu_impl;
mod deconv2d_cpu_impl;

use self::{
    conv2d_bckwd_filter_cpu::conv2d_bckwd_fileter, conv2d_cpu_impl::conv2d_inner,
    deconv2d_cpu_impl::deconv2d_inner,
};

#[cfg(feature = "nvidia")]
use zenu_cuda::{
    cudnn::{conv::{ConvolutionBuilder, ConvolutionConfig, backward_bias}, TensorFormat},
    kernel::conv_bias_add,
};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

#[must_use]
pub fn conv2d_out_size(
    img_shape: &[usize],
    kernel_shape: &[usize],
    padding: (usize, usize),
    stride: (usize, usize),
) -> [usize; 4] {
    let (b, h, w) = (img_shape[0], img_shape[2], img_shape[3]);
    let (oc, kh, kw) = (kernel_shape[0], kernel_shape[2], kernel_shape[3]);
    let (ph, pw) = padding;
    let (sh, sw) = stride;
    let (h, w) = ((h + 2 * ph - kh) / sh + 1, (w + 2 * pw - kw) / sw + 1);
    [b, oc, h, w]
}

pub(super) fn get_deconv_outsize_(size: usize, k: usize, s: usize, p: usize) -> usize {
    s * (size - 1) + k - 2 * p
}

#[must_use]
pub fn deconv2d_out_size(
    img_shape: &[usize],
    kernel_shape: &[usize],
    padding: (usize, usize),
    stride: (usize, usize),
) -> [usize; 4] {
    let (b, h, w) = (img_shape[0], img_shape[2], img_shape[3]);
    let (ic, kh, kw) = (kernel_shape[1], kernel_shape[2], kernel_shape[3]);
    let (ph, pw) = padding;
    let (sh, sw) = stride;
    let (h, w) = (
        get_deconv_outsize_(h, kh, sh, ph),
        get_deconv_outsize_(w, kw, sw, pw),
    );
    [b, ic, h, w]
}

pub struct Conv2dConfig<T: Num> {
    #[cfg(feature = "nvidia")]
    pub conv: ConvolutionConfig<T>,
    _phantom: std::marker::PhantomData<T>,
}

#[expect(clippy::too_many_arguments, clippy::missing_panics_doc)]
#[must_use]
pub fn create_conv_descriptor<T: Num>(
    input_shape: &[usize],
    output_shape: &[usize],
    filter_shape: &[usize],
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
) -> Conv2dConfig<T> {
    #[cfg(feature = "nvidia")]
    let conv = {
        let input_shape = input_shape.iter().map(|x| i32::try_from(*x).unwrap()).collect::<Vec<_>>();
        let output_shape = output_shape.iter().map(|x| i32::try_from(*x).unwrap()).collect::<Vec<_>>();
        let filter_shape = filter_shape.iter().map(|x| i32::try_from(*x).unwrap()).collect::<Vec<_>>();

        let input_shape_0: i32 = input_shape[0] ;
        let input_shape_1: i32 = input_shape[1] ;
        let input_shape_2: i32 = input_shape[2] ;
        let input_shape_3: i32 = input_shape[3] ;

        let output_shape_0: i32 = output_shape[0] ;
        let output_shape_1: i32 = output_shape[1] ;
        let output_shape_2: i32 = output_shape[2] ;
        let output_shape_3: i32 = output_shape[3] ;

        let filter_shape_0: i32 = filter_shape[0] ;
        let filter_shape_1: i32 = filter_shape[1] ;
        let filter_shape_2: i32 = filter_shape[2] ;
        let filter_shape_3: i32 = filter_shape[3] ;

        let pad_h: i32 = pad_h.try_into().unwrap();
        let pad_w: i32 = pad_w.try_into().unwrap();

        let stride_h: i32 = stride_h.try_into().unwrap(); 
        let stride_w: i32 = stride_w.try_into().unwrap();

        let dilation_h: i32 = dilation_h.try_into().unwrap();
        let dilation_w: i32 = dilation_w.try_into().unwrap(); 

        let conv = ConvolutionBuilder::<T>::default()
            .input(
                input_shape_0,
                input_shape_1,
                input_shape_2,
                input_shape_3,
                TensorFormat::NCHW,
            )
            .unwrap()
            .filter(
                filter_shape_0,
                filter_shape_1,
                filter_shape_2,
                filter_shape_3,
                TensorFormat::NCHW,
            )
            .unwrap()
            .output(
                output_shape_0,
                output_shape_1,
                output_shape_2,
                output_shape_3,
                TensorFormat::NCHW,
            )
            .unwrap()
            .conv(pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w)
            .unwrap()
            .algorithms(groups);

        conv.build()
    };

    Conv2dConfig {
        #[cfg(feature = "nvidia")]
        conv,
        _phantom: std::marker::PhantomData,
    }
}

pub trait Conv2d: DeviceBase {
    #[expect(clippy::too_many_arguments)]
    fn conv2d<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        filter: Matrix<Ref<&T>, DimDyn, Self>,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        config: Option<&Conv2dConfig<T>>,
    );

    #[expect(clippy::too_many_arguments)]
    fn conv2d_bckwd_data<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        dx: Matrix<Ref<&mut T>, DimDyn, Self>,
        filter: Matrix<Ref<&T>, DimDyn, Self>,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        config: Option<&Conv2dConfig<T>>,
    );

    #[expect(clippy::too_many_arguments)]
    fn conv2d_bckwd_filter<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        df: Matrix<Ref<&mut T>, DimDyn, Self>,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        config: Option<&Conv2dConfig<T>>,
    );

    fn conv2d_forward_bias<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
    );

    fn conv2d_bckwd_bias<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        dx: Matrix<Ref<&mut T>, DimDyn, Self>,
    );
}

impl Conv2d for Cpu {
    fn conv2d<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        filter: Matrix<Ref<&T>, DimDyn, Self>,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        _config: Option<&Conv2dConfig<T>>,
    ) {
        if dilation_h != 1 || dilation_w != 1 {
            todo!();
        }
        y.copy_from(&conv2d_inner(
            input,
            filter,
            None,
            (pad_h, pad_w),
            (stride_h, stride_w),
        ));
    }

    fn conv2d_bckwd_data<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        dx: Matrix<Ref<&mut T>, DimDyn, Self>,
        filter: Matrix<Ref<&T>, DimDyn, Self>,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        _config: Option<&Conv2dConfig<T>>,
    ) {
        if dilation_h != 1 || dilation_w != 1 {
            todo!();
        }
        dx.copy_from(&deconv2d_inner(
            dy,
            filter,
            None,
            (pad_h, pad_w),
            (stride_h, stride_w),
        ));
    }

    fn conv2d_bckwd_filter<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        df: Matrix<Ref<&mut T>, DimDyn, Self>,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        _config: Option<&Conv2dConfig<T>>,
    ) {
        if dilation_h != 1 || dilation_w != 1 {
            todo!();
        }
        df.copy_from(&conv2d_bckwd_fileter(
            input,
            df.to_ref().shape(),
            dy,
            (pad_h, pad_w),
            (stride_h, stride_w),
        ));
    }

    fn conv2d_forward_bias<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        mut y: Matrix<Ref<&mut T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
    ) {
        y.add_array(&input, &bias);
    }

    fn conv2d_bckwd_bias<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        dx: Matrix<Ref<&mut T>, DimDyn, Self>,
    ) {
        let dy_0 = dy.sum(0, true);
        let dy_0_2 = dy_0.to_ref().sum(2, true);
        dx.copy_from(&dy_0_2.to_ref().sum(3, true));
    }
}

#[cfg(feature = "nvidia")]
impl Conv2d for Nvidia {
    fn conv2d<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        filter: Matrix<Ref<&T>, DimDyn, Self>,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        config: Option<&Conv2dConfig<T>>,
    ) {
        let config = match config {
            Some(config) => &config.conv,
            None => {
                &create_conv_descriptor::<T>(
                    input.shape().slice(),
                    y.shape().slice(),
                    filter.shape().slice(),
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    1,
                )
                .conv
            }
        };

        config.forward(
            T::one(),
            input.as_ptr(),
            filter.as_ptr(),
            T::zero(),
            y.as_mut_ptr(),
        );
    }

    fn conv2d_bckwd_data<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        dx: Matrix<Ref<&mut T>, DimDyn, Self>,
        filter: Matrix<Ref<&T>, DimDyn, Self>,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        config: Option<&Conv2dConfig<T>>,
    ) {
        let config = match config {
            Some(config) => &config.conv,
            None => {
                &create_conv_descriptor::<T>(
                    dx.shape().slice(),
                    dy.shape().slice(),
                    filter.shape().slice(),
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    1,
                )
                .conv
            }
        };

        config.backward_data(
            T::one(),
            filter.as_ptr(),
            dy.as_ptr(),
            T::zero(),
            dx.as_mut_ptr(),
        );
    }

    fn conv2d_bckwd_filter<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        df: Matrix<Ref<&mut T>, DimDyn, Self>,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        config: Option<&Conv2dConfig<T>>,
    ) {
        let config = match config {
            Some(config) => &config.conv,
            None => {
                &create_conv_descriptor::<T>(
                    input.shape().slice(),
                    dy.shape().slice(),
                    df.shape().slice(),
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    1,
                )
                .conv
            }
        };

        config.backward_filter(
            T::one(),
            input.as_ptr(),
            dy.as_ptr(),
            T::zero(),
            df.as_mut_ptr(),
        );
    }

    fn conv2d_forward_bias<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&mut T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
    ) {
        let input_channel_stride = input.stride()[1];
        let input_num_elm = input.shape().num_elm();
        let bias_num_elm = bias.shape().num_elm();

        conv_bias_add(
            input.as_ptr(),
            bias.as_ptr(),
            input_num_elm,
            input_channel_stride,
            bias_num_elm,
            y.as_mut_ptr(),
        );
    }

    fn conv2d_bckwd_bias<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        dx: Matrix<Ref<&mut T>, DimDyn, Self>,
    ) {
        let dy_shape = dy.shape();
        let dy_shape = dy_shape.slice();
        backward_bias(T::one(), dy.as_ptr(), T::zero(), dx.as_mut_ptr(), dy_shape);
    }
}

#[expect(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_forward<T: Num, D: Device>(
    input: Matrix<Ref<&T>, DimDyn, D>,
    filter: Matrix<Ref<&T>, DimDyn, D>,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    config: Option<&Conv2dConfig<T>>,
) -> Matrix<Owned<T>, DimDyn, D> {
    let out_size = conv2d_out_size(
        input.shape().slice(),
        filter.shape().slice(),
        (pad_h, pad_w),
        (stride_h, stride_w),
    );
    let mut y = Matrix::<Owned<T>, DimDyn, D>::alloc(out_size);
    D::conv2d(
        input,
        y.to_ref_mut(),
        filter,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        config,
    );
    y
}

#[expect(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_bckwd_data<T: Num, D: Device>(
    dy: Matrix<Ref<&T>, DimDyn, D>,
    filter: Matrix<Ref<&T>, DimDyn, D>,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    config: Option<&Conv2dConfig<T>>,
) -> Matrix<Owned<T>, DimDyn, D> {
    let input_shape = deconv2d_out_size(
        dy.shape().slice(),
        filter.shape().slice(),
        (pad_h, pad_w),
        (stride_h, stride_w),
    );
    let mut dx = Matrix::<Owned<T>, DimDyn, D>::alloc(input_shape);
    D::conv2d_bckwd_data(
        dy,
        dx.to_ref_mut(),
        filter,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        config,
    );
    dx
}

#[expect(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d_bckwd_filter<T: Num, D: Device>(
    input: Matrix<Ref<&T>, DimDyn, D>,
    dy: Matrix<Ref<&T>, DimDyn, D>,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    filter_shape: DimDyn,
    config: Option<&Conv2dConfig<T>>,
) -> Matrix<Owned<T>, DimDyn, D> {
    let mut df = Matrix::<Owned<T>, DimDyn, D>::alloc(filter_shape);
    D::conv2d_bckwd_filter(
        input,
        dy,
        df.to_ref_mut(),
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        config,
    );
    df
}

pub fn conv2d_bias_add<T: Num, D: Device>(
    input: Matrix<Ref<&T>, DimDyn, D>,
    bias: Matrix<Ref<&T>, DimDyn, D>,
    output: Matrix<Ref<&mut T>, DimDyn, D>,
) {
    D::conv2d_forward_bias(input, output, bias);
}

pub fn conv2d_bckwd_data_bias<T: Num, D: Device>(
    dy: Matrix<Ref<&T>, DimDyn, D>,
    dx: Matrix<Ref<&mut T>, DimDyn, D>,
) {
    D::conv2d_bckwd_bias(dy, dx);
}

#[expect(clippy::unreadable_literal, clippy::too_many_lines)]
#[cfg(test)]
mod conv2d {
    use zenu_test::{assert_mat_eq_epsilon, run_mat_test};

    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    use super::{conv2d_bckwd_data, conv2d_bckwd_filter, conv2d_forward};

    #[expect(dead_code)]
    struct Conv2dTestCase<D: Device> {
        input: Matrix<Owned<f32>, DimDyn, D>,
        filter: Matrix<Owned<f32>, DimDyn, D>,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        expected: Matrix<Owned<f32>, DimDyn, D>,
        input_grad: Matrix<Owned<f32>, DimDyn, D>,
        filter_grad: Matrix<Owned<f32>, DimDyn, D>,
        output_grad: Matrix<Owned<f32>, DimDyn, D>,
    }

    fn conv2d_test<D: Device>() {
        let test_case = small::<D>();

        let forward_pred = conv2d_forward(
            test_case.input.to_ref(),
            test_case.filter.to_ref(),
            test_case.pad_h,
            test_case.pad_w,
            test_case.stride_h,
            test_case.stride_w,
            test_case.dilation_h,
            test_case.dilation_w,
            None,
        );
        assert_mat_eq_epsilon!(forward_pred, test_case.expected, 1e-4);
        let input_grad = conv2d_bckwd_data(
            test_case.output_grad.to_ref(),
            test_case.filter.to_ref(),
            test_case.pad_h,
            test_case.pad_w,
            test_case.stride_h,
            test_case.stride_w,
            test_case.dilation_h,
            test_case.dilation_w,
            None,
        );
        assert_mat_eq_epsilon!(input_grad, test_case.input_grad, 1e-4);

        let filter_grad = conv2d_bckwd_filter(
            test_case.input.to_ref(),
            test_case.output_grad.to_ref(),
            test_case.pad_h,
            test_case.pad_w,
            test_case.stride_h,
            test_case.stride_w,
            test_case.dilation_h,
            test_case.dilation_w,
            test_case.filter.shape(),
            None,
        );
        assert_mat_eq_epsilon!(filter_grad, test_case.filter_grad, 1e-4);
    }
    run_mat_test!(conv2d_test, conv2d_test_cpu, conv2d_test_nvidia);

    fn small<D: Device>() -> Conv2dTestCase<D> {
        let input = vec![
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
        let output = vec![
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
        let input_grad = vec![
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
        let filter = vec![
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
        let filter_grad = vec![
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
        let input = Matrix::<Owned<f32>, DimDyn, D>::from_vec(input, [1, 2, 4, 4]);
        let input_grad = Matrix::<Owned<f32>, DimDyn, D>::from_vec(input_grad, [1, 2, 4, 4]);
        let filter = Matrix::<Owned<f32>, DimDyn, D>::from_vec(filter, [3, 2, 3, 3]);
        let filter_grad = Matrix::<Owned<f32>, DimDyn, D>::from_vec(filter_grad, [3, 2, 3, 3]);
        let expected = Matrix::<Owned<f32>, DimDyn, D>::from_vec(output, [1, 3, 4, 4]);
        let output_grad = Matrix::<Owned<f32>, DimDyn, D>::ones([1, 3, 4, 4]);

        Conv2dTestCase {
            input,
            filter,
            pad_h: 1,
            pad_w: 1,
            stride_h: 1,
            stride_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            expected,
            input_grad,
            filter_grad,
            output_grad,
        }
    }
}

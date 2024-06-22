use crate::{
    device::{cpu::Cpu, Device, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref},
    num::Num,
};

mod col2im;
mod conv2d_bckwd_filter_cpu;
mod conv2d_cpu_impl;
mod deconv2d_cpu_impl;
mod im2col;

#[cfg(feature = "nvidia")]
use zenu_cuda::cudnn::{conv::*, TensorFormat};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

use self::{
    conv2d_bckwd_filter_cpu::conv2d_bckwd_fileter,
    conv2d_cpu_impl::conv2d_inner,
    deconv2d_cpu_impl::{deconv2d_inner, deconv2d_out_size},
};

macro_rules! impl_conv_config {
    ($name:ident, $inner:ident, $inner_builder:ident, $desc_create:ident) => {
        pub struct $name<T: Num> {
            #[cfg(feature = "nvidia")]
            pub conv: $inner,
            _phantom: std::marker::PhantomData<T>,
        }

        #[cfg(feature = "nvidia")]
        fn $desc_create<T: Num>(
            input: DimDyn,
            output: DimDyn,
            filter: DimDyn,
            pad_h: usize,
            pad_w: usize,
            stride_h: usize,
            stride_w: usize,
            dilation_h: usize,
            dilation_w: usize,
            num_algo: usize,
        ) -> $inner {
            println!(
                "input: {:?}, output: {:?}, filter: {:?}, pad_h: {}, pad_w: {}, stride_h: {}, stride_w: {}, dilation_h: {}, dilation_w: {}, num_algo: {}",
                input, output, filter, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, num_algo
            );
            $inner_builder::default()
                .input::<T>(
                    input[0].try_into().unwrap(),
                    input[1].try_into().unwrap(),
                    input[2].try_into().unwrap(),
                    input[3].try_into().unwrap(),
                    TensorFormat::NCHW,
                )
                .unwrap()
                .filter::<T>(
                    filter[0].try_into().unwrap(),
                    filter[1].try_into().unwrap(),
                    filter[2].try_into().unwrap(),
                    filter[3].try_into().unwrap(),
                    TensorFormat::NCHW,
                )
                .unwrap()
                .output::<T>(
                    output[0].try_into().unwrap(),
                    output[1].try_into().unwrap(),
                    output[2].try_into().unwrap(),
                    output[3].try_into().unwrap(),
                    TensorFormat::NCHW,
                )
                .unwrap()
                .conv(
                    pad_h.try_into().unwrap(),
                    pad_w.try_into().unwrap(),
                    stride_h.try_into().unwrap(),
                    stride_w.try_into().unwrap(),
                    dilation_h.try_into().unwrap(),
                    dilation_w.try_into().unwrap(),
                )
                .unwrap()
                .algorithm(num_algo)
                .unwrap()
                .build()
                .unwrap()
        }

        impl<T: Num> $name<T> {
            pub fn new(
                input: DimDyn,
                output: DimDyn,
                filter: DimDyn,
                pad_h: usize,
                pad_w: usize,
                stride_h: usize,
                stride_w: usize,
                dilation_h: usize,
                dilation_w: usize,
                num_algo: usize,
            ) -> Self {
                Self {
                    #[cfg(feature = "nvidia")]
                    conv: $desc_create::<T>(
                        input, output, filter, pad_h, pad_w, stride_h, stride_w, dilation_h,
                        dilation_w, num_algo,
                    ),
                    _phantom: std::marker::PhantomData,
                }
            }
        }
    };
}
impl_conv_config!(
    Conv2dConfig,
    ConvDescriptor,
    ConvolutionBuilder,
    create_conv_descriptor
);
impl_conv_config!(
    Conv2dBckwdDataConfig,
    ConvolutionBackwardData,
    ConvolutionBackwardDataBuilder,
    create_conv_bckwd_data
);
impl_conv_config!(
    Conv2dBckwdFilterConfig,
    ConvolutionBackwardFilter,
    ConvolutionBackwardFilterBuilder,
    create_conv_bckwd_filter
);

pub trait Conv2d: DeviceBase {
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
        config: Option<Conv2dConfig<T>>,
    );

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
        config: Option<Conv2dBckwdDataConfig<T>>,
    );

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
        config: Option<Conv2dBckwdFilterConfig<T>>,
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
        _config: Option<Conv2dConfig<T>>,
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
        _config: Option<Conv2dBckwdDataConfig<T>>,
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
        _config: Option<Conv2dBckwdFilterConfig<T>>,
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
        config: Option<Conv2dConfig<T>>,
    ) {
        let config = match config {
            Some(config) => config.conv,
            None => create_conv_descriptor::<T>(
                input.shape(),
                y.shape(),
                filter.shape(),
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                1,
            ),
        };

        config.forward(
            T::one(),
            input.as_ptr(),
            filter.as_ptr(),
            T::zero(),
            y.as_mut_ptr(),
        )
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
        config: Option<Conv2dBckwdDataConfig<T>>,
    ) {
        let config = match config {
            Some(config) => config.conv,
            None => create_conv_bckwd_data::<T>(
                dy.shape(),
                dx.shape(),
                filter.shape(),
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                1,
            ),
        };

        config.backward_data(
            T::one(),
            dy.as_ptr(),
            filter.as_ptr(),
            T::zero(),
            dx.as_mut_ptr(),
        )
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
        config: Option<Conv2dBckwdFilterConfig<T>>,
    ) {
        let config = match config {
            Some(config) => config.conv,
            None => create_conv_bckwd_filter::<T>(
                dy.shape(),
                input.shape(),
                df.shape(),
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                1,
            ),
        };

        config.backward_filter(
            T::one(),
            input.as_ptr(),
            dy.as_ptr(),
            T::zero(),
            df.as_mut_ptr(),
        )
    }
}

pub fn conv2d_forward<T: Num, D: Device>(
    input: Matrix<Ref<&T>, DimDyn, D>,
    filter: Matrix<Ref<&T>, DimDyn, D>,
    bias: Option<Matrix<Ref<&T>, DimDyn, D>>,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    config: Option<Conv2dConfig<T>>,
) -> Matrix<Owned<T>, DimDyn, D> {
    let out_size = conv2d_cpu_impl::conv2d_out_size(
        input.shape().slice(),
        filter.shape().slice(),
        (pad_h, pad_w),
        (stride_h, stride_w),
    );
    let mut y = Matrix::<Owned<T>, DimDyn, D>::zeros(out_size);
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
    if let Some(bias) = bias {
        y += bias;
    }
    y
}

pub fn conv2d_bckwd_data<T: Num, D: Device>(
    dy: Matrix<Ref<&T>, DimDyn, D>,
    filter: Matrix<Ref<&T>, DimDyn, D>,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    config: Option<Conv2dBckwdDataConfig<T>>,
) -> Matrix<Owned<T>, DimDyn, D> {
    let input_shape = deconv2d_out_size(
        dy.shape().slice(),
        filter.shape().slice(),
        (pad_h, pad_w),
        (stride_h, stride_w),
    );
    let mut dx = Matrix::<Owned<T>, DimDyn, D>::zeros(input_shape);
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
    config: Option<Conv2dBckwdFilterConfig<T>>,
) -> Matrix<Owned<T>, DimDyn, D> {
    let mut df = Matrix::<Owned<T>, DimDyn, D>::zeros(filter_shape);
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

#[cfg(test)]
mod conv2d {
    use zenu_test::{assert_mat_eq_epsilon, run_mat_test};

    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
        slice_dynamic,
    };

    use super::{conv2d_bckwd_data, conv2d_bckwd_filter, conv2d_forward};

    struct Conv2dTestCase<D: Device> {
        input: Matrix<Owned<f32>, DimDyn, D>,
        filter: Matrix<Owned<f32>, DimDyn, D>,
        bias: Option<Matrix<Owned<f32>, DimDyn, D>>,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        expected: Matrix<Owned<f32>, DimDyn, D>,
        input_grad: Matrix<Owned<f32>, DimDyn, D>,
        filter_grad: Matrix<Owned<f32>, DimDyn, D>,
        bias_grad: Option<Matrix<Owned<f32>, DimDyn, D>>,
        output_grad: Matrix<Owned<f32>, DimDyn, D>,
    }

    fn conv2d_test<D: Device>() {
        let test_case = small::<D>();

        let forward_pred = conv2d_forward(
            test_case.input.to_ref(),
            test_case.filter.to_ref(),
            None,
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
            test_case.filter.shape().clone(),
            None,
        );
        assert_mat_eq_epsilon!(filter_grad, test_case.filter_grad, 1e-4);
    }
    run_mat_test!(conv2d_test, conv2d_test_cpu, conv2d_test_nvidia);

    fn small<D: Device>() -> Conv2dTestCase<D> {
        let kernl = (1..(4 * 3 * 3 * 3 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let kernl = Matrix::<Owned<f32>, DimDyn, D>::from_vec(kernl, &[4, 3, 3, 3]);
        let image = (1..(2 * 3 * 5 * 5 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let image = Matrix::<Owned<f32>, DimDyn, D>::from_vec(image, &[2, 3, 5, 5]);
        let output_ans = vec![
            7416., 11010., 11289., 11568., 7608., 11106., 16434., 16812., 17190., 11268., 12411.,
            18324., 18702., 19080., 12483., 13716., 20214., 20592., 20970., 13698., 8712., 12792.,
            13017., 13242., 8616., 16812., 25347., 26112., 26877., 17976., 26415., 39762., 40869.,
            41976., 28035., 30150., 45297., 46404., 47511., 31680., 33885., 50832., 51939., 53046.,
            35325., 22968., 34419., 35130., 35841., 23844., 26208., 39684., 40935., 42186., 28344.,
            41724., 63090., 64926., 66762., 44802., 47889., 72270., 74106., 75942., 50877., 54054.,
            81450., 83286., 85122., 56952., 37224., 56046., 57243., 58440., 39072., 35604., 54021.,
            55758., 57495., 38712., 57033., 86418., 88983., 91548., 61569., 65628., 99243.,
            101808., 104373., 70074., 74223., 112068., 114633., 117198., 78579., 51480., 77673.,
            79356., 81039., 54300., 21816., 31935., 32214., 32493., 21108., 30681., 44784., 45162.,
            45540., 29493., 31986., 46674., 47052., 47430., 30708., 33291., 48564., 48942., 49320.,
            31923., 20412., 29667., 29892., 30117., 19416., 55512., 82722., 83487., 84252., 55776.,
            82440., 122787., 123894., 125001., 82710., 86175., 128322., 129429., 130536., 86355.,
            89910., 133857., 134964., 136071., 90000., 58968., 87744., 88455., 89166., 58944.,
            89208., 133509., 134760., 136011., 90444., 134199., 200790., 202626., 204462., 135927.,
            140364., 209970., 211806., 213642., 142002., 146529., 219150., 220986., 222822.,
            148077., 97524., 145821., 147018., 148215., 98472., 122904., 184296., 186033., 187770.,
            125112., 185958., 278793., 281358., 283923., 189144., 194553., 291618., 294183.,
            296748., 197649., 203148., 304443., 307008., 309573., 206154., 136080., 203898.,
            205581., 207264., 138000.,
        ];
        let output_ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(output_ans, &[2, 4, 5, 5]);

        let kernel_grad_ans = vec![
            1520., 1920., 1552., 2000., 2525., 2040., 1680., 2120., 1712., 2320., 2920., 2352.,
            3000., 3775., 3040., 2480., 3120., 2512., 3120., 3920., 3152., 4000., 5025., 4040.,
            3280., 4120., 3312., 1520., 1920., 1552., 2000., 2525., 2040., 1680., 2120., 1712.,
            2320., 2920., 2352., 3000., 3775., 3040., 2480., 3120., 2512., 3120., 3920., 3152.,
            4000., 5025., 4040., 3280., 4120., 3312., 1520., 1920., 1552., 2000., 2525., 2040.,
            1680., 2120., 1712., 2320., 2920., 2352., 3000., 3775., 3040., 2480., 3120., 2512.,
            3120., 3920., 3152., 4000., 5025., 4040., 3280., 4120., 3312., 1520., 1920., 1552.,
            2000., 2525., 2040., 1680., 2120., 1712., 2320., 2920., 2352., 3000., 3775., 3040.,
            2480., 3120., 2512., 3120., 3920., 3152., 4000., 5025., 4040., 3280., 4120., 3312.,
        ];
        let kernel_grad_ans =
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(kernel_grad_ans, &[4, 3, 3, 3]);

        let image_grad_ans = vec![
            696., 1056., 1056., 1056., 712., 1080., 1638., 1638., 1638., 1104., 1080., 1638.,
            1638., 1638., 1104., 1080., 1638., 1638., 1638., 1104., 744., 1128., 1128., 1128.,
            760., 840., 1272., 1272., 1272., 856., 1296., 1962., 1962., 1962., 1320., 1296., 1962.,
            1962., 1962., 1320., 1296., 1962., 1962., 1962., 1320., 888., 1344., 1344., 1344.,
            904., 984., 1488., 1488., 1488., 1000., 1512., 2286., 2286., 2286., 1536., 1512.,
            2286., 2286., 2286., 1536., 1512., 2286., 2286., 2286., 1536., 1032., 1560., 1560.,
            1560., 1048., 696., 1056., 1056., 1056., 712., 1080., 1638., 1638., 1638., 1104.,
            1080., 1638., 1638., 1638., 1104., 1080., 1638., 1638., 1638., 1104., 744., 1128.,
            1128., 1128., 760., 840., 1272., 1272., 1272., 856., 1296., 1962., 1962., 1962., 1320.,
            1296., 1962., 1962., 1962., 1320., 1296., 1962., 1962., 1962., 1320., 888., 1344.,
            1344., 1344., 904., 984., 1488., 1488., 1488., 1000., 1512., 2286., 2286., 2286.,
            1536., 1512., 2286., 2286., 2286., 1536., 1512., 2286., 2286., 2286., 1536., 1032.,
            1560., 1560., 1560., 1048.,
        ];
        let image_grad_ans =
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(image_grad_ans, &[2, 3, 5, 5]);

        Conv2dTestCase {
            input: image,
            filter: kernl,
            bias: None,
            pad_h: 1,
            pad_w: 1,
            stride_h: 1,
            stride_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            expected: output_ans,
            input_grad: image_grad_ans,
            filter_grad: kernel_grad_ans,
            bias_grad: None,
            output_grad: Matrix::<Owned<f32>, DimDyn, D>::from_vec(
                vec![1.; 2 * 4 * 5 * 5],
                &[2, 4, 5, 5],
            ),
        }
    }
}

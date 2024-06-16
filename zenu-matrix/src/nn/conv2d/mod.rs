use crate::{
    device::{cpu::Cpu, DeviceBase},
    dim::DimDyn,
    matrix::{Matrix, Ref},
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
    conv2d_bckwd_filter_cpu::conv2d_bckwd_fileter, conv2d_cpu_impl::conv2d_inner,
    deconv2d_cpu_impl::deconv2d_inner,
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
            $inner_builder::default()
                .input::<T>(
                    input[0].try_into().unwrap(),
                    input[1].try_into().unwrap(),
                    input[2].try_into().unwrap(),
                    input[3].try_into().unwrap(),
                    TensorFormat::NHWC,
                )
                .unwrap()
                .filter::<T>(
                    filter[0].try_into().unwrap(),
                    filter[1].try_into().unwrap(),
                    filter[2].try_into().unwrap(),
                    filter[3].try_into().unwrap(),
                    TensorFormat::NHWC,
                )
                .unwrap()
                .output::<T>(
                    output[0].try_into().unwrap(),
                    output[1].try_into().unwrap(),
                    output[2].try_into().unwrap(),
                    output[3].try_into().unwrap(),
                    TensorFormat::NHWC,
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
            df.to_ref(),
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
                input.shape(),
                dy.shape(),
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

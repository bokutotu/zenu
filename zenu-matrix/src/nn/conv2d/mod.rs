use crate::{
    device::DeviceBase,
    dim::DimDyn,
    matrix::{Matrix, Ref},
    num::Num,
};

pub mod col2im;
pub mod conv2d_cpu_impl;
pub mod deconv2d_cpu_impl;
pub mod im2col;

#[cfg(feature = "nvidia")]
use zenu_cuda::cudnn::{conv::*, TensorFormat};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

pub struct Conv2dConfig<T: Num> {
    #[cfg(feature = "nvidia")]
    pub conv: ConvDescriptor,
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "nvidia")]
fn create_conv_descriptor<T: Num>(
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
) -> ConvDescriptor {
    ConvolutionBuilder::default()
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

impl<T: Num> Conv2dConfig<T> {
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
            conv: create_conv_descriptor::<T>(
                input, output, filter, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                num_algo,
            ),
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct Conv2dBckwdDataConfig<T: Num> {
    #[cfg(feature = "nvidia")]
    pub conv: ConvolutionBackwardData,
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "nvidia")]
fn create_conv_bckwd_data<T: Num>(
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
) -> ConvolutionBackwardData {
    ConvolutionBackwardDataBuilder::default()
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

impl<T: Num> Conv2dBckwdDataConfig<T> {
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
            conv: create_conv_bckwd_data::<T>(
                input, output, filter, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                num_algo,
            ),
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct Conv2dBckwdFilterConfig<T: Num> {
    #[cfg(feature = "nvidia")]
    pub conv: ConvolutionBackwardFilter,
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "nvidia")]
fn create_conv_bckwd_filter<T: Num>(
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
) -> ConvolutionBackwardFilter {
    ConvolutionBackwardFilterBuilder::default()
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

impl<T: Num> Conv2dBckwdFilterConfig<T> {
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
            conv: create_conv_bckwd_filter::<T>(
                input, output, filter, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                num_algo,
            ),
            _phantom: std::marker::PhantomData,
        }
    }
}

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
        config: Conv2dConfig<T>,
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
        config: Conv2dBckwdDataConfig<T>,
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
        config: Conv2dBckwdFilterConfig<T>,
    );
}

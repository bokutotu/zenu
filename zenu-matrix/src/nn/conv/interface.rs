#[cfg(feature = "nvidia")]
use zenu_cuda::cudnn::graph_conv::{
    ConvBkwdDataGraph, ConvBkwdFilterGraph, ConvBuilder, ConvForwardGraph,
};

#[cfg(feature = "nvidia")]
use crate::dim::DimTrait;

use crate::{
    device::DeviceBase,
    dim::{default_stride, DimDyn},
    matrix::{Matrix, Ref},
    num::Num,
    shape_stride::ShapeStride,
};

use super::utils::conv_output_shape;

#[derive(Clone, Debug, PartialEq)]
pub struct ConvConfigInner<T> {
    pub input_shape: ShapeStride<DimDyn>,
    pub filter_shape: ShapeStride<DimDyn>,
    pub output_shape: ShapeStride<DimDyn>,
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
    pub dilation: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

pub struct ConvFwdConfig<T> {
    pub inner: ConvConfigInner<T>,
    #[cfg(feature = "nvidia")]
    pub nvidia_desc: ConvForwardGraph,
}

pub trait ConvFwd: DeviceBase {
    fn conv_fwd<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        weight: Matrix<Ref<&T>, DimDyn, Self>,
        output: Matrix<Ref<&mut T>, DimDyn, Self>,
        config: &mut ConvFwdConfig<T>,
    );
}

pub struct ConvBkwdDataConfig<T> {
    pub inner: ConvConfigInner<T>,
    #[cfg(feature = "nvidia")]
    pub nvidia_desc: ConvBkwdDataGraph,
}

pub trait ConvBkwdData: DeviceBase {
    fn conv_bkwd_data<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        filter: Matrix<Ref<&T>, DimDyn, Self>,
        dx: Matrix<Ref<&mut T>, DimDyn, Self>,
        config: &mut ConvBkwdDataConfig<T>,
    );
}

pub struct ConvBkwdFilterConfig<T> {
    pub inner: ConvConfigInner<T>,
    #[cfg(feature = "nvidia")]
    pub nvidia_desc: ConvBkwdFilterGraph,
}

pub trait ConvBkwdFilter: DeviceBase {
    fn conv_bkwd_filter<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        dw: Matrix<Ref<&mut T>, DimDyn, Self>,
        config: &mut ConvBkwdFilterConfig<T>,
    );
}

fn build_conv_inner<T>(
    input_shape: ShapeStride<DimDyn>,
    filter_shape: ShapeStride<DimDyn>,
    stride: Vec<usize>,
    padding: Vec<usize>,
    dilation: Vec<usize>,
) -> ConvConfigInner<T> {
    // convolutionの入力次元はconv1dの場合は3, conv2dの場合は4
    // つまり2を引くとconvolutionの次元数になる
    let conv_dim = input_shape.len() - 2;
    assert_eq!(
        stride.len(),
        conv_dim,
        "Stride length must match the number of spatial dimensions."
    );
    assert_eq!(
        padding.len(),
        conv_dim,
        "Padding length must match the number of spatial dimensions."
    );
    assert_eq!(
        dilation.len(),
        conv_dim,
        "Dilation length must match the number of spatial dimensions."
    );
    let output_shape = conv_output_shape(
        input_shape.shape(),
        filter_shape.shape(),
        &stride,
        &padding,
        &dilation,
    );

    let output_stride = default_stride(output_shape);
    ConvConfigInner {
        input_shape,
        filter_shape,
        output_shape: ShapeStride::new(output_shape, output_stride),
        stride,
        padding,
        dilation,
        _phantom: std::marker::PhantomData,
    }
}

impl<T> ConvFwdConfig<T> {
    #[must_use]
    pub fn new(
        input_shape: ShapeStride<DimDyn>,
        filter_shape: ShapeStride<DimDyn>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
    ) -> Self {
        let inner = build_conv_inner(input_shape, filter_shape, stride, padding, dilation);
        let res: Self = inner.into();
        res
    }
}

#[cfg(feature = "nvidia")]
fn conv_builder<T>(inner: &ConvConfigInner<T>) -> ConvBuilder<T> {
    ConvBuilder::<T>::default()
        .x_shape(inner.input_shape.shape().slice().to_vec())
        .x_stride(inner.input_shape.stride().slice().to_vec())
        .w_shape(inner.filter_shape.shape().slice().to_vec())
        .y_shape(inner.output_shape.shape().slice().to_vec())
        .y_stride(inner.output_shape.stride().slice().to_vec())
        .stride(inner.stride.clone())
        .pad(inner.padding.clone())
        .dilation(inner.dilation.clone())
}

#[cfg(feature = "nvidia")]
fn build_conv_fwd_graph<T>(inner: &ConvConfigInner<T>) -> ConvForwardGraph {
    conv_builder(inner).build_forward()
}

#[cfg(feature = "nvidia")]
fn build_conv_bkwd_data_graph<T>(inner: &ConvConfigInner<T>) -> ConvBkwdDataGraph {
    conv_builder(inner).build_bkwd_data()
}

#[cfg(feature = "nvidia")]
fn build_conv_bkwd_filter_graph<T>(inner: &ConvConfigInner<T>) -> ConvBkwdFilterGraph {
    conv_builder(inner).build_bkwd_filter()
}

impl<T> From<ConvConfigInner<T>> for ConvFwdConfig<T> {
    fn from(inner: ConvConfigInner<T>) -> Self {
        #[cfg(feature = "nvidia")]
        let nvidia_desc = build_conv_fwd_graph(&inner);
        Self {
            inner,
            #[cfg(feature = "nvidia")]
            nvidia_desc,
        }
    }
}

impl<T> From<ConvConfigInner<T>> for ConvBkwdDataConfig<T> {
    fn from(inner: ConvConfigInner<T>) -> Self {
        #[cfg(feature = "nvidia")]
        let nvidia_desc = build_conv_bkwd_data_graph(&inner);
        Self {
            inner,
            #[cfg(feature = "nvidia")]
            nvidia_desc,
        }
    }
}

impl<T> From<ConvConfigInner<T>> for ConvBkwdFilterConfig<T> {
    fn from(inner: ConvConfigInner<T>) -> Self {
        #[cfg(feature = "nvidia")]
        let nvidia_desc = build_conv_bkwd_filter_graph(&inner);
        Self {
            inner,
            #[cfg(feature = "nvidia")]
            nvidia_desc,
        }
    }
}

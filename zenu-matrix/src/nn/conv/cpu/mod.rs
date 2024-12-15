use conv_fwd::conv_fwd;

use crate::{
    device::cpu::Cpu,
    dim::DimDyn,
    matrix::{Matrix, Ref},
    num::Num,
};

use super::interface::{
    ConvBkwdData, ConvBkwdDataConfig, ConvBkwdFilter, ConvBkwdFilterConfig, ConvFwd, ConvFwdConfig,
};

pub(super) mod col2im;
mod conv_bkwd_data;
mod conv_bkwd_filter;
mod conv_fwd;
pub(super) mod im2col;

impl ConvFwd for Cpu {
    fn conv_fwd<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        weight: Matrix<Ref<&T>, DimDyn, Self>,
        output: Matrix<Ref<&mut T>, DimDyn, Self>,
        _: &mut ConvFwdConfig<T>,
    ) {
        let n = input.shape()[0];
        let c_in = input.shape()[1];
        let h_in = input.shape()[2];
        let w_in = input.shape()[3];
        let c_out = weight.shape()[0];
        let kh = weight.shape()[2];
        let kw = weight.shape()[3];
        let pad_h = 0;
        let pad_w = 0;
        let stride_h = 1;
        let stride_w = 1;
        let dilation_h = 1;
        let dilation_w = 1;

        conv_fwd(
            input.as_slice_unchecked(),
            weight.as_slice_unchecked(),
            n,
            c_in,
            h_in,
            w_in,
            c_out,
            kh,
            kw,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            output.as_mut_slice_unchecked(),
        );
    }
}

impl ConvBkwdData for Cpu {
    fn conv_bkwd_data<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        filter: Matrix<Ref<&T>, DimDyn, Self>,
        dx: Matrix<Ref<&mut T>, DimDyn, Self>,
        _: &mut ConvBkwdDataConfig<T>,
    ) {
        let n = dx.shape()[0];
        let c_in = dx.shape()[1];
        let h_in = dx.shape()[2];
        let w_in = dx.shape()[3];
        let c_out = filter.shape()[0];
        let kh = filter.shape()[2];
        let kw = filter.shape()[3];
        let pad_h = 0;
        let pad_w = 0;
        let stride_h = 1;
        let stride_w = 1;
        let dilation_h = 1;
        let dilation_w = 1;

        conv_bkwd_data::conv_bkwd_data(
            dy.as_slice_unchecked(),
            filter.as_slice_unchecked(),
            n,
            c_in,
            c_out,
            h_in,
            w_in,
            kh,
            kw,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            dx.as_mut_slice_unchecked(),
        );
    }
}

impl ConvBkwdFilter for Cpu {
    fn conv_bkwd_filter<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        dw: Matrix<Ref<&mut T>, DimDyn, Self>,
        _: &mut ConvBkwdFilterConfig<T>,
    ) {
        let n = x.shape()[0];
        let c_in = x.shape()[1];
        let h_in = x.shape()[2];
        let w_in = x.shape()[3];
        let c_out = dy.shape()[1];
        let kh = dw.shape()[2];
        let kw = dw.shape()[3];
        let pad_h = 0;
        let pad_w = 0;
        let stride_h = 1;
        let stride_w = 1;
        let dilation_h = 1;
        let dilation_w = 1;

        conv_bkwd_filter::conv_bkwd_filter(
            dy.as_slice_unchecked(),
            x.as_slice_unchecked(),
            n,
            c_in,
            c_out,
            h_in,
            w_in,
            kh,
            kw,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            dw.as_mut_slice_unchecked(),
        );
    }
}

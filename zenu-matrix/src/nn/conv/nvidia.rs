use zenu_cuda::kernel::{conv_bias_add, conv_bias_bkwd};

use crate::{
    device::nvidia::Nvidia,
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Ref},
    nn::NNCache,
    num::Num,
};

use super::interface::{
    ConvBias, ConvBkwdData, ConvBkwdDataConfig, ConvBkwdFilter, ConvBkwdFilterConfig,
    ConvConfigInner, ConvFwd, ConvFwdConfig,
};

impl ConvFwd for Nvidia {
    fn conv_fwd<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        weight: Matrix<Ref<&T>, DimDyn, Self>,
        output: Matrix<Ref<&mut T>, DimDyn, Self>,
        config: &mut ConvFwdConfig<T>,
    ) {
        let now_inner = ConvConfigInner::new(
            input.shape_stride(),
            weight.shape_stride(),
            output.shape_stride(),
            &config.inner.padding,
            &config.inner.stride,
            &config.inner.dilation,
        );
        assert!(config.inner.valid(&now_inner), "Invalid ConvFwdConfig.");

        if config.inner.is_batch_size_changed(input.shape()[0]) {
            let new_config = now_inner.into();
            let _ = std::mem::replace(config, new_config);
        }

        let workspace_size = config.nvidia_desc.get_workspace_size();
        let workspace = NNCache::<Self>::new(workspace_size);

        config.nvidia_desc.execute(
            input.as_ptr().cast_mut(),
            weight.as_ptr().cast_mut(),
            output.as_mut_ptr(),
            workspace.ptr.cast(),
        );
    }
}

impl ConvBkwdData for Nvidia {
    fn conv_bkwd_data<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        filter: Matrix<Ref<&T>, DimDyn, Self>,
        dx: Matrix<Ref<&mut T>, DimDyn, Self>,
        config: &mut ConvBkwdDataConfig<T>,
    ) {
        let now_inner = ConvConfigInner::new(
            dx.shape_stride(),
            filter.shape_stride(),
            dy.shape_stride(),
            &config.inner.padding,
            &config.inner.stride,
            &config.inner.dilation,
        );
        assert!(
            config.inner.valid(&now_inner),
            "Invalid ConvBkwdDataConfig."
        );

        if config.inner.is_batch_size_changed(dy.shape()[0]) {
            let new_config = now_inner.into();
            let _ = std::mem::replace(config, new_config);
        }

        let workspace_size = config.nvidia_desc.get_workspace_size();
        let workspace = NNCache::<Self>::new(workspace_size);

        config.nvidia_desc.execute(
            dy.as_ptr().cast_mut(),
            filter.as_ptr().cast_mut(),
            dx.as_mut_ptr(),
            workspace.ptr.cast(),
        );
    }
}

impl ConvBkwdFilter for Nvidia {
    fn conv_bkwd_filter<T: Num>(
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        dw: Matrix<Ref<&mut T>, DimDyn, Self>,
        config: &mut ConvBkwdFilterConfig<T>,
    ) {
        let now_inner = ConvConfigInner::new(
            x.shape_stride(),
            dw.shape_stride(),
            dy.shape_stride(),
            &config.inner.padding,
            &config.inner.stride,
            &config.inner.dilation,
        );
        assert!(
            config.inner.valid(&now_inner),
            "Invalid ConvBkwdFilterConfig."
        );

        if config.inner.is_batch_size_changed(dy.shape()[0]) {
            let new_config = now_inner.into();
            let _ = std::mem::replace(config, new_config);
        }

        let workspace_size = config.nvidia_desc.get_workspace_size();
        let workspace = NNCache::<Self>::new(workspace_size);

        config.nvidia_desc.execute(
            dy.as_ptr().cast_mut(),
            x.as_ptr().cast_mut(),
            dw.as_mut_ptr(),
            workspace.ptr.cast(),
        );
    }
}

impl ConvBias for Nvidia {
    fn conv2d_bias<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
        output: Matrix<Ref<&mut T>, DimDyn, Self>,
    ) {
        let input_channel_stride = input.stride()[1];
        let input_num_elm = input.shape().num_elm();
        let bias_num_elm = bias.shape().num_elm();

        conv_bias_add(
            input.as_ptr().cast(),
            bias.as_ptr().cast(),
            input_num_elm,
            input_channel_stride,
            bias_num_elm,
            output.as_mut_ptr(),
        );
    }

    fn conv2d_bias_bkwd<T: Num>(
        d_output: Matrix<Ref<&T>, DimDyn, Self>,
        bias: Matrix<Ref<&mut T>, DimDyn, Self>,
    ) {
        let n = d_output.shape()[0];
        let c = d_output.shape()[1];
        let h = d_output.shape()[2];
        let w = d_output.shape()[3];

        conv_bias_bkwd(d_output.as_ptr(), bias.as_mut_ptr(), n, c, h, w);
    }
}

use crate::{
    device::nvidia::Nvidia,
    dim::DimDyn,
    matrix::{Matrix, Ref},
    nn::NNCache,
    num::Num,
};

use super::interface::{
    ConvBkwdData, ConvBkwdDataConfig, ConvBkwdFilter, ConvBkwdFilterConfig, ConvConfigInner,
    ConvFwd, ConvFwdConfig,
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

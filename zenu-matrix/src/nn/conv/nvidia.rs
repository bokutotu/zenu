use crate::{
    device::nvidia::Nvidia,
    dim::DimDyn,
    matrix::{Matrix, Ref},
    nn::NNCache,
    num::Num,
};

use super::interface::{ConvConfigInner, ConvFwd, ConvFwdConfig};

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

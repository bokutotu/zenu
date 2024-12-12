use std::ptr::NonNull;

use zenu_cudnn_frontend_wrapper_sys::{
    check_graph, create_batch_norm_descriptor, execute_batch_norm_forward_training,
    get_workspace_size, BatchNormBkwdDescriptor, BatchNormDescriptor, BatchNormExecutionBuffers,
};
use zenu_cudnn_sys::cudnnHandle_t;

use crate::ZENU_CUDA_STATE;

use super::graph_utils::{get_cudnn_frontend_type, shape_stride_to_cudnn, success_or_panic};

pub struct BatchNormForward<T> {
    ptr: NonNull<BatchNormDescriptor>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: 'static> BatchNormForward<T> {
    #[must_use]
    pub fn new(
        shape: &[usize],
        stride: &[usize],
        epsilon: f32,
        momentum: f32,
        is_training: bool,
    ) -> Self {
        let mut desc = std::ptr::null_mut();
        let data_type = get_cudnn_frontend_type::<T>();
        let shape_stride = shape_stride_to_cudnn(shape, stride);
        let status = unsafe {
            create_batch_norm_descriptor(
                &mut desc,
                data_type,
                std::ptr::from_ref(&shape_stride),
                epsilon,
                momentum,
                is_training,
            )
        };
        success_or_panic(status);
        Self {
            ptr: NonNull::new(desc).unwrap(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn check_and_build_graph(&self) {
        let mut handle: cudnnHandle_t = ZENU_CUDA_STATE.lock().unwrap().get_cudnn_handle();
        let status =
            unsafe { check_graph(self.ptr.as_ptr(), std::ptr::from_mut(&mut handle).cast()) };
        success_or_panic(status);
    }

    #[must_use]
    pub fn get_workspace_size(&self) -> usize {
        let mut size = 0;
        let status = unsafe { get_workspace_size(self.ptr.as_ptr(), &mut size) };
        success_or_panic(status);
        usize::try_from(size).unwrap()
    }

    #[expect(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        x: *const T,
        mean: *mut T,
        inv_variance: *mut T,
        scale: *mut T,
        bias: *mut T,
        peer_stats_0: *mut T,
        peer_stats_1: *mut T,
        prev_running_mean: *mut T,
        prev_running_var: *mut T,
        next_running_mean: *mut T,
        next_running_var: *mut T,
        y: *mut T,
        workspace: *mut u8,
    ) {
        let mut buffers = BatchNormExecutionBuffers {
            X: x.cast_mut().cast(),
            mean: mean.cast(),
            inv_variance: inv_variance.cast(),
            scale: scale.cast(),
            bias: bias.cast(),
            peer_stats_0: peer_stats_0.cast(),
            peer_stats_1: peer_stats_1.cast(),
            prev_running_mean: prev_running_mean.cast(),
            prev_running_var: prev_running_var.cast(),
            next_running_mean: next_running_mean.cast(),
            next_running_var: next_running_var.cast(),
            Y: y.cast(),
        };
        let mut handle: cudnnHandle_t = ZENU_CUDA_STATE.lock().unwrap().get_cudnn_handle();
        let status = unsafe {
            execute_batch_norm_forward_training(
                self.ptr.as_ptr(),
                std::ptr::from_mut(&mut buffers),
                workspace.cast(),
                std::ptr::from_mut(&mut handle).cast(),
            )
        };
        success_or_panic(status);
    }
}

//! 実験的なものなのでまだ使用できない
use std::ptr::NonNull;

use zenu_cudnn_frontend_wrapper_sys::{
    check_backward_data_graph, check_graph, create_batch_norm_backward_data_descriptor,
    create_batch_norm_descriptor, execute_batch_norm_backward_data,
    execute_batch_norm_forward_training, get_backward_data_workspace_size, get_workspace_size,
    BatchNormBkwdDescriptor, BatchNormBkwdExecutionBuffers, BatchNormDescriptor,
    BatchNormExecutionBuffers,
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

pub struct BatchNormBkwd<T> {
    ptr: NonNull<BatchNormBkwdDescriptor>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: 'static> BatchNormBkwd<T> {
    #[must_use]
    pub fn new(shape: &[usize], stride: &[usize]) -> Self {
        let mut desc = std::ptr::null_mut();
        let data_type = get_cudnn_frontend_type::<T>();
        let shape_stride = shape_stride_to_cudnn(shape, stride);
        let status = unsafe {
            create_batch_norm_backward_data_descriptor(
                &mut desc,
                data_type,
                std::ptr::from_ref(&shape_stride),
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
        let status = unsafe {
            check_backward_data_graph(self.ptr.as_ptr(), std::ptr::from_mut(&mut handle).cast())
        };
        success_or_panic(status);
    }

    #[must_use]
    pub fn get_workspace_size(&self) -> usize {
        let mut size = 0;
        let status = unsafe { get_backward_data_workspace_size(self.ptr.as_ptr(), &mut size) };
        success_or_panic(status);
        usize::try_from(size).unwrap()
    }

    #[expect(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        x: *const T,
        dy: *const T,
        scale: *const T,
        mean: *const T,
        inv_variance: *const T,
        dscale: *mut T,
        dbias: *mut T,
        dx: *mut T,
        peer_stats_0: *const T,
        peer_stats_1: *const T,
        workspace: *mut u8,
    ) {
        let mut buffers = BatchNormBkwdExecutionBuffers {
            X: x.cast_mut().cast(),
            DY: dy.cast_mut().cast(),
            scale: scale.cast_mut().cast(),
            mean: mean.cast_mut().cast(),
            inv_variance: inv_variance.cast_mut().cast(),
            dscale: dscale.cast(),
            dbias: dbias.cast(),
            DX: dx.cast(),
            peer_stats_0: peer_stats_0.cast_mut().cast(),
            peer_stats_1: peer_stats_1.cast_mut().cast(),
        };
        let mut handle: cudnnHandle_t = ZENU_CUDA_STATE.lock().unwrap().get_cudnn_handle();
        let status = unsafe {
            execute_batch_norm_backward_data(
                self.ptr.as_ptr(),
                std::ptr::from_mut(&mut buffers),
                workspace.cast(),
                std::ptr::from_mut(&mut handle).cast(),
            )
        };
        success_or_panic(status);
    }
}

// #[cfg(test)]
// mod batchnorm_test {
//     use crate::runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind};
//
//     use super::BatchNormForward;
//
//     #[test]
//     #[expect(clippy::unreadable_literal)]
//     fn batchnorm_test_case_1() {
//         let n = 2;
//         let c = 24;
//         let h = 22;
//         let w = 25;
//         let shape = [n, c, h, w];
//         let stride = [c * h * w, h * w, w, 1];
//         let stats_shape = [2, c * 4, 1, 1];
//
//         let batchnorm_fwd = BatchNormForward::<f32>::new(&shape, &stride, 1e-5, 0., true);
//         println!("here: ");
//         batchnorm_fwd.check_and_build_graph();
//         let workspace_size = batchnorm_fwd.get_workspace_size();
//         let input_cpu = [
//             -1.1258398,
//             -1.1523602,
//             -0.25057858,
//             -0.4338788,
//             0.84871036,
//             0.69200915,
//             -0.31601277,
//             -2.1152194,
//             0.32227492,
//             -1.2633348,
//             0.3499832,
//             0.30813393,
//             0.11984151,
//             1.2376579,
//             1.1167772,
//             -0.24727815,
//         ];
//         let output_cpu = [
//             -1.0970649,
//             -1.1374662,
//             0.23631285,
//             -0.04292771,
//             0.66504365,
//             0.5121599,
//             -0.4713051,
//             -2.2266803,
//             1.109001,
//             -1.3065253,
//             1.1512119,
//             1.0874585,
//             -0.04606889,
//             1.0445158,
//             0.92657995,
//             -0.40424496,
//         ];
//         let running_mean = [-0.04057, 0.01670607];
//         let running_variance = [0.9492437, 1.0200632];
//         let saved_mean = [-0.04057, 0.01670607];
//         let saved_variance = [0.9492437, 1.0200632];
//         let scale = [1.0, 1.0];
//         let bias = [0.0, 0.0];
//
//         let input_gpu = cpu_vec_to_gpu(&input_cpu);
//         let output_gpu = cuda_malloc(output_cpu.len()).unwrap();
//         let running_mean_gpu = cpu_vec_to_gpu(&running_mean);
//         let running_variance_gpu = cpu_vec_to_gpu(&running_variance);
//         let saved_mean_gpu = cpu_vec_to_gpu(&saved_mean);
//         let saved_variance_gpu = cpu_vec_to_gpu(&saved_variance);
//         let scale_gpu = cpu_vec_to_gpu(&scale);
//         let bias_gpu = cpu_vec_to_gpu(&bias);
//         let workspace_gpu = cuda_malloc::<u8>(workspace_size).unwrap();
//         let peer_stats_0_gpu = cuda_malloc::<f32>(stats_shape.iter().product()).unwrap();
//         let peer_stats_1_gpu = cuda_malloc::<f32>(stats_shape.iter().product()).unwrap();
//         let next_running_mean_gpu = cuda_malloc::<f32>(running_mean.len()).unwrap();
//         let next_running_var_gpu = cuda_malloc::<f32>(running_variance.len()).unwrap();
//
//         batchnorm_fwd.execute(
//             input_gpu,
//             running_mean_gpu,
//             running_variance_gpu,
//             scale_gpu,
//             bias_gpu,
//             peer_stats_0_gpu,
//             peer_stats_1_gpu,
//             saved_mean_gpu,
//             saved_variance_gpu,
//             next_running_mean_gpu,
//             next_running_var_gpu,
//             output_gpu,
//             workspace_gpu,
//         );
//
//         let output_exp = gpu_to_cpu_vec(output_gpu, output_cpu.len());
//         for i in 0..output_cpu.len() h            assert!((output_cpu[i] - output_exp[i]).abs() < 1e-6);
//         }
//     }
//
//     fn cpu_vec_to_gpu<T: 'static>(vec: &[T]) -> *mut T {
//         let gpu = cuda_malloc(vec.len()).unwrap();
//         cuda_copy(
//             gpu,
//             vec.as_ptr(),
//             vec.len(),
//             ZenuCudaMemCopyKind::HostToDevice,
//         )
//         .unwrap();
//         gpu
//     }
//
//     fn gpu_to_cpu_vec<T: 'static + Default + Clone>(gpu: *const T, len: usize) -> Vec<T> {
//         let mut vec = vec![T::default(); len];
//         cuda_copy(
//             vec.as_mut_ptr(),
//             gpu,
//             len,
//             ZenuCudaMemCopyKind::DeviceToHost,
//         )
//         .unwrap();
//         vec
//     }
// }

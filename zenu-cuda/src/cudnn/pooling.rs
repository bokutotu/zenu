use zenu_cudnn_sys::{
    cudnnCreatePoolingDescriptor, cudnnDestroyPoolingDescriptor, cudnnDestroyTensorDescriptor,
    cudnnNanPropagation_t, cudnnPoolingBackward, cudnnPoolingDescriptor_t, cudnnPoolingForward,
    cudnnPoolingMode_t, cudnnSetPooling2dDescriptor, cudnnStatus_t, cudnnTensorDescriptor_t,
};

use crate::ZENU_CUDA_STATE;

use super::{error::ZenuCudnnError, tensor_descriptor_4d, TensorFormat};

fn pooling_descriptor(
    mode: cudnnPoolingMode_t,
    window_h: usize,
    window_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<cudnnPoolingDescriptor_t, ZenuCudnnError> {
    let mut pooling: cudnnPoolingDescriptor_t = std::ptr::null_mut();
    unsafe {
        // let status = cudnnCreatePoolingDescriptor(&mut pooling as *mut cudnnPoolingDescriptor_t);
        let status = cudnnCreatePoolingDescriptor(std::ptr::from_mut(&mut pooling));
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        let status = cudnnSetPooling2dDescriptor(
            pooling,
            mode,
            cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            i32::try_from(window_h).unwrap(),
            i32::try_from(window_w).unwrap(),
            i32::try_from(pad_h).unwrap(),
            i32::try_from(pad_w).unwrap(),
            i32::try_from(stride_h).unwrap(),
            i32::try_from(stride_w).unwrap(),
        );
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(pooling)
}

pub struct Pool2d<T> {
    pooling: cudnnPoolingDescriptor_t,
    input_desc: cudnnTensorDescriptor_t,
    output_desc: cudnnTensorDescriptor_t,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Drop for Pool2d<T> {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyPoolingDescriptor(self.pooling);
            cudnnDestroyTensorDescriptor(self.input_desc);
            cudnnDestroyTensorDescriptor(self.output_desc);
        }
    }
}

pub enum PoolType {
    Max,
    Average,
    AverageCountExcludePadding,
    MaxDeterministic,
}

impl From<PoolType> for cudnnPoolingMode_t {
    fn from(pool_type: PoolType) -> Self {
        match pool_type {
            PoolType::Max => cudnnPoolingMode_t::CUDNN_POOLING_MAX,
            PoolType::Average => cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            PoolType::AverageCountExcludePadding => {
                cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
            }
            PoolType::MaxDeterministic => cudnnPoolingMode_t::CUDNN_POOLING_MAX_DETERMINISTIC,
        }
    }
}

impl<T: 'static + Copy> Pool2d<T> {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        pool_type: PoolType,
        window_h: usize,
        window_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        input_shape: (usize, usize, usize, usize),
        output_shape: (usize, usize, usize, usize),
    ) -> Result<Self, ZenuCudnnError> {
        let pooling = pooling_descriptor(
            pool_type.into(),
            window_h,
            window_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
        )?;
        let input_desc = tensor_descriptor_4d::<T>(
            input_shape.0.try_into().unwrap(),
            input_shape.1.try_into().unwrap(),
            input_shape.2.try_into().unwrap(),
            input_shape.3.try_into().unwrap(),
            TensorFormat::NCHW,
        )?;
        let output_desc = tensor_descriptor_4d::<T>(
            output_shape.0.try_into().unwrap(),
            output_shape.1.try_into().unwrap(),
            output_shape.2.try_into().unwrap(),
            output_shape.3.try_into().unwrap(),
            TensorFormat::NCHW,
        )?;
        Ok(Self {
            pooling,
            input_desc,
            output_desc,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn forward(
        &self,
        input: *const T,
        output: *mut T,
        alpha: T,
        beta: T,
    ) -> Result<(), ZenuCudnnError> {
        let handle = ZENU_CUDA_STATE.lock().unwrap();
        let handle = handle.get_cudnn();
        let status = unsafe {
            cudnnPoolingForward(
                handle.as_ptr(),
                self.pooling as cudnnPoolingDescriptor_t,
                std::ptr::from_ref(&alpha).cast::<std::ffi::c_void>(),
                self.input_desc,
                input.cast(),
                std::ptr::from_ref(&beta).cast::<std::ffi::c_void>(),
                self.output_desc,
                output.cast(),
            )
        };
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        Ok(())
    }

    pub fn backward(
        &self,
        input: *const T,
        input_grad: *mut T,
        output: *const T,
        output_grad: *const T,
        alpha: T,
        beta: T,
    ) -> Result<(), ZenuCudnnError> {
        let handle = ZENU_CUDA_STATE.lock().unwrap();
        let handle = handle.get_cudnn();
        let status = unsafe {
            cudnnPoolingBackward(
                handle.as_ptr(),
                self.pooling as cudnnPoolingDescriptor_t,
                std::ptr::from_ref(&alpha).cast::<std::ffi::c_void>(),
                self.output_desc,
                output.cast(),
                self.output_desc,
                output_grad.cast(),
                self.input_desc,
                input.cast(),
                std::ptr::from_ref(&beta).cast::<std::ffi::c_void>(),
                self.input_desc,
                input_grad.cast(),
            )
        };

        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
        Ok(())
    }
}

#[cfg(test)]
mod pool2d {
    use crate::runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind};

    use super::Pool2d;

    #[expect(clippy::too_many_lines)]
    #[test]
    fn max_pool_2d() {
        let input = vec![
            -1.1258398,
            -1.1523602,
            -0.25057858,
            -0.4338788,
            0.84871036,
            0.69200915,
            -0.31601277,
            -2.1152194,
            0.32227492,
            -1.2633348,
            0.3499832,
            0.30813393,
            0.11984151,
            1.2376579,
            1.1167772,
            -0.24727815,
            -1.3526537,
            -1.6959312,
            0.5666506,
            0.79350835,
            0.59883946,
            -1.5550951,
            -0.3413604,
            1.8530061,
            0.7501895,
            -0.58549756,
            -0.17339675,
            0.18347794,
            1.3893661,
            1.5863342,
            0.94629836,
            -0.84367675,
            -0.6135831,
            0.03159274,
            -0.49267697,
            0.24841475,
            0.43969584,
            0.112411186,
            0.64079237,
            0.44115627,
            -0.10230965,
            0.792444,
            -0.2896677,
            0.052507486,
            0.52286047,
            2.3022053,
            -1.4688939,
            -1.5866888,
            -0.6730899,
            0.8728312,
            1.0553575,
            0.17784372,
            -0.23033547,
            -0.3917544,
            0.5432947,
            -0.39515755,
            -0.44621718,
            0.7440207,
            1.5209795,
            2.3803675,
            -1.1256016,
            -0.3169981,
            -1.0924683,
            -0.0851943,
            -0.093348235,
            0.6870502,
            -0.83831537,
            0.018486667,
            -0.7504268,
            0.18540798,
            0.62113833,
            0.63818157,
            -0.24600095,
            2.3025165,
            -1.8816892,
        ];
        let output = [
            0.69200915, 1.2376579, 0.59883946, 1.8530061, 0.94629836, 1.5863342, 2.3022053,
            0.8728312, 1.0553575, 2.3803675, 0.6870502, 2.3025165,
        ];

        // let input_descrptor = tensor_descriptor_4d::<f32>(1, 3, 5, 5, TensorFormat::NCHW).unwrap();
        // let output_descrptor = tensor_descriptor_4d::<f32>(1, 3, 2, 2, TensorFormat::NCHW).unwrap();
        let pool = Pool2d::<f32>::new(
            super::PoolType::Max,
            3,
            3,
            0,
            0,
            2,
            2,
            (1, 3, 5, 5),
            (1, 3, 2, 2),
        )
        .unwrap();
        let input_gpu = vec_to_gpu(&input);
        let output_gpu = cuda_malloc::<f32>(output.len()).unwrap();

        pool.forward(input_gpu, output_gpu, 1.0, 0.0).unwrap();

        let output_cudnn = gpu_to_vec(output_gpu, output.len());
        for (a, b) in output.iter().zip(output_cudnn.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        let output_grad = vec![1.0; output.len()];
        let output_grad_gpu = vec_to_gpu(&output_grad);

        let input_grad_gpu = cuda_malloc::<f32>(input.len()).unwrap();

        pool.backward(
            input_gpu,
            input_grad_gpu,
            output_gpu,
            output_grad_gpu,
            1.0,
            0.0,
        )
        .unwrap();

        let input_grad = gpu_to_vec(input_grad_gpu, input.len());
        let input_grad_ans = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];

        for (a, b) in input_grad.iter().zip(input_grad_ans.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    fn vec_to_gpu(vec: &[f32]) -> *mut f32 {
        let gpu_ptr = cuda_malloc::<f32>(vec.len()).unwrap();
        cuda_copy(
            gpu_ptr,
            vec.as_ptr(),
            vec.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        gpu_ptr
    }

    fn gpu_to_vec(gpu_ptr: *const f32, len: usize) -> Vec<f32> {
        let mut vec = vec![0.0; len];
        cuda_copy(
            vec.as_mut_ptr(),
            gpu_ptr,
            len,
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();
        vec
    }
}

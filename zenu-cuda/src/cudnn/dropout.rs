use zenu_cudnn_sys::{
    cudnnCreateDropoutDescriptor, cudnnDestroyDropoutDescriptor, cudnnDestroyTensorDescriptor,
    cudnnDropoutBackward, cudnnDropoutDescriptor_t, cudnnDropoutForward,
    cudnnDropoutGetReserveSpaceSize, cudnnDropoutGetStatesSize, cudnnHandle_t,
    cudnnSetDropoutDescriptor, cudnnStatus_t, cudnnTensorDescriptor_t,
};

use crate::ZENU_CUDA_STATE;

use super::{error::ZenuCudnnError, tensor_descriptor_2d, tensor_descriptor_4d};

pub(crate) fn dropout_descriptor() -> Result<cudnnDropoutDescriptor_t, ZenuCudnnError> {
    let mut dropout: cudnnDropoutDescriptor_t = std::ptr::null_mut();
    unsafe {
        let status = cudnnCreateDropoutDescriptor(std::ptr::from_mut(&mut dropout));
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(dropout)
}

fn destory_dropout_descriptor(desc: cudnnDropoutDescriptor_t) -> Result<(), ZenuCudnnError> {
    unsafe {
        let status = cudnnDestroyDropoutDescriptor(desc);
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(())
}

fn dropout_set(
    dropout_desc: cudnnDropoutDescriptor_t,
    dropout: f32,
    seed: u64,
    state: *mut std::ffi::c_void,
    state_size_in_bytes: usize,
) -> Result<(), ZenuCudnnError> {
    unsafe {
        let status = cudnnSetDropoutDescriptor(
            dropout_desc,
            ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr(),
            dropout,
            state,
            state_size_in_bytes,
            seed,
        );
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(())
}

fn dropout_forward<T: 'static>(
    handle: cudnnHandle_t,
    dropout_desc: cudnnDropoutDescriptor_t,
    x: *const T,
    y: *mut T,
    reserve_space: *mut std::ffi::c_void,
    reserve_space_size_in_bytes: usize,
    tensor_desc: cudnnTensorDescriptor_t,
) -> Result<(), ZenuCudnnError> {
    unsafe {
        let status = cudnnDropoutForward(
            handle,
            dropout_desc,
            tensor_desc,
            x.cast(),
            tensor_desc,
            y.cast(),
            reserve_space,
            reserve_space_size_in_bytes,
        );
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(())
}

fn dropout_backward<T: 'static>(
    handle: cudnnHandle_t,
    dropout_desc: cudnnDropoutDescriptor_t,
    dy: *const T,
    dx: *mut T,
    reserve_space: *mut std::ffi::c_void,
    reserve_space_size_in_bytes: usize,
    tensor_desc: cudnnTensorDescriptor_t,
) -> Result<(), ZenuCudnnError> {
    unsafe {
        let status = cudnnDropoutBackward(
            handle,
            dropout_desc,
            tensor_desc,
            dy.cast(),
            tensor_desc,
            dx.cast(),
            reserve_space,
            reserve_space_size_in_bytes,
        );
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(())
}

fn dropout_reserve_space_size(
    tensor_desc: cudnnTensorDescriptor_t,
    size_in_bytes: *mut usize,
) -> Result<(), ZenuCudnnError> {
    unsafe {
        let status = cudnnDropoutGetReserveSpaceSize(tensor_desc, size_in_bytes);
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(())
}

fn dropout_state_size() -> Result<i32, ZenuCudnnError> {
    let mut size_in_bytes = 0;
    unsafe {
        let status = cudnnDropoutGetStatesSize(
            ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr(),
            &mut size_in_bytes,
        );
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(ZenuCudnnError::from(status));
        }
    }
    Ok(i32::try_from(size_in_bytes).unwrap())
}

pub struct DropoutConfig<T: 'static> {
    dropout_desc: cudnnDropoutDescriptor_t,
    tensor_desc: cudnnTensorDescriptor_t,
    reserve_space_size: usize,
    state_size: i32,
    _maker: std::marker::PhantomData<T>,
}

impl<T: 'static> DropoutConfig<T> {
    pub fn new(shape: &[usize]) -> Result<Self, ZenuCudnnError> {
        let shape = shape
            .iter()
            .map(|x| i32::try_from(*x).unwrap())
            .collect::<Vec<i32>>();
        let tensor_desc = if shape.len() == 4 {
            tensor_descriptor_4d::<T>(
                shape[0],
                shape[1],
                shape[2],
                shape[3],
                super::TensorFormat::NCHW,
            )?
        } else if shape.len() == 2 {
            tensor_descriptor_2d::<T>(shape[0], shape[1])?
        } else {
            panic!("shape");
        };
        let dropout_desc = dropout_descriptor()?;
        let reserve_space_size = {
            let mut size = 0;
            dropout_reserve_space_size(tensor_desc, &mut size)?;
            size
        };
        let state_size = dropout_state_size()?;
        Ok(Self {
            dropout_desc,
            tensor_desc,
            reserve_space_size,
            state_size,
            _maker: std::marker::PhantomData,
        })
    }

    #[must_use]
    pub fn get_reserve_space_size(&self) -> usize {
        self.reserve_space_size
    }

    #[must_use]
    pub fn get_state_size(&self) -> usize {
        usize::try_from(self.state_size).unwrap()
    }

    pub fn set(
        &self,
        dropout: f32,
        seed: u64,
        state: *mut std::ffi::c_void,
    ) -> Result<(), ZenuCudnnError> {
        dropout_set(
            self.dropout_desc,
            dropout,
            seed,
            state,
            self.get_state_size(),
        )
    }

    pub fn forward(
        &self,
        x: *const T,
        y: *mut T,
        reserve_space: *mut std::ffi::c_void,
    ) -> Result<(), ZenuCudnnError> {
        let handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();
        dropout_forward(
            handle,
            self.dropout_desc,
            x,
            y,
            reserve_space,
            self.reserve_space_size,
            self.tensor_desc,
        )
    }

    pub fn backward(
        &self,
        dy: *const T,
        dx: *mut T,
        reserve_space: *mut std::ffi::c_void,
    ) -> Result<(), ZenuCudnnError> {
        let handle = ZENU_CUDA_STATE.lock().unwrap().get_cudnn().as_ptr();
        dropout_backward(
            handle,
            self.dropout_desc,
            dy,
            dx,
            reserve_space,
            self.reserve_space_size,
            self.tensor_desc,
        )
    }
}

impl<T: 'static> Drop for DropoutConfig<T> {
    fn drop(&mut self) {
        destory_dropout_descriptor(self.dropout_desc).unwrap();
        unsafe { cudnnDestroyTensorDescriptor(self.tensor_desc) };
    }
}

#[cfg(test)]
mod dropout_test {
    use crate::runtime::{cuda_copy, cuda_malloc, cuda_malloc_bytes, ZenuCudaMemCopyKind};

    use super::*;

    #[expect(clippy::similar_names)]
    #[test]
    fn test_dropout_4d() {
        let dropout = DropoutConfig::<f32>::new(&[2, 3, 2, 2]).unwrap();
        let space_size = dropout.get_reserve_space_size();
        let state_size = dropout.get_state_size();
        let space = cuda_malloc_bytes(space_size).unwrap();
        let state = cuda_malloc::<u8>(state_size).unwrap();
        dropout.set(0.5, 0, state.cast()).unwrap();
        let input_cpu = [
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ];
        let mut output_cpu = vec![0.; 24];
        let input_gpu = cuda_malloc::<f32>(24).unwrap();
        let output_gpu = cuda_malloc::<f32>(24).unwrap();
        cuda_copy(
            input_gpu,
            input_cpu.as_ptr(),
            24,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        dropout
            .forward(input_gpu, output_gpu, space.cast())
            .unwrap();
        cuda_copy(
            output_cpu.as_mut_ptr(),
            output_gpu,
            24,
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();
        let input_grad_expect = output_cpu
            .iter()
            .map(|x| if *x == 0. { 0. } else { 1. / 0.5 })
            .collect::<Vec<f32>>();

        let output_grad_cpu = [1.0; 24];
        let mut input_grad_cpu = vec![0.0; 24];
        let output_grad_gpu = cuda_malloc::<f32>(24).unwrap();
        let input_grad_gpu = cuda_malloc::<f32>(24).unwrap();
        cuda_copy(
            output_grad_gpu,
            output_grad_cpu.as_ptr(),
            24,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        dropout
            .backward(output_grad_gpu, input_grad_gpu, space.cast())
            .unwrap();

        cuda_copy(
            input_grad_cpu.as_mut_ptr(),
            input_grad_gpu,
            24,
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();

        assert_eq!(input_grad_expect, input_grad_cpu);
    }

    // #[test]
    // fn test_dropout_2d() {
    //     let dropout = DropoutConfig::<f32>::new(&[2 * 3, 2 * 2]).unwrap();
    //     let space_size = dropout.get_reserve_space_size();
    //     let state_size = dropout.get_state_size();
    //     let space = cuda_malloc_bytes(space_size).unwrap();
    //     let state = cuda_malloc::<u8>(state_size as usize).unwrap();
    //     dropout.set(0.5, 0, state as *mut ::libc::c_void).unwrap();
    //     let input_cpu = vec![
    //         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
    //         20., 21., 22., 23., 24.,
    //     ];
    //     let mut output_cpu = vec![0.; 24];
    //     let input_gpu = cuda_malloc::<f32>(24).unwrap();
    //     let output_gpu = cuda_malloc::<f32>(24).unwrap();
    //     cuda_copy(
    //         input_gpu,
    //         input_cpu.as_ptr(),
    //         24,
    //         ZenuCudaMemCopyKind::HostToDevice,
    //     )
    //     .unwrap();
    //
    //     dropout
    //         .forward(input_gpu, output_gpu, space as *mut ::libc::c_void)
    //         .unwrap();
    //     cuda_copy(
    //         output_cpu.as_mut_ptr(),
    //         output_gpu,
    //         24,
    //         ZenuCudaMemCopyKind::DeviceToHost,
    //     )
    //     .unwrap();
    //     let input_grad_expect = output_cpu
    //         .iter()
    //         .map(|x| if *x == 0. { 0. } else { 1. / 0.5 })
    //         .collect::<Vec<f32>>();
    //
    //     let output_grad_cpu = vec![1.0; 24];
    //     let mut input_grad_cpu = vec![0.0; 24];
    //     let output_grad_gpu = cuda_malloc::<f32>(24).unwrap();
    //     let input_grad_gpu = cuda_malloc::<f32>(24).unwrap();
    //     cuda_copy(
    //         output_grad_gpu,
    //         output_grad_cpu.as_ptr(),
    //         24,
    //         ZenuCudaMemCopyKind::HostToDevice,
    //     )
    //     .unwrap();
    //
    //     dropout
    //         .backward(
    //             output_grad_gpu,
    //             input_grad_gpu,
    //             space as *mut ::libc::c_void,
    //         )
    //         .unwrap();
    //
    //     cuda_copy(
    //         input_grad_cpu.as_mut_ptr(),
    //         input_grad_gpu,
    //         24,
    //         ZenuCudaMemCopyKind::DeviceToHost,
    //     )
    //     .unwrap();
    //
    //     assert_eq!(input_grad_expect, input_grad_cpu);
    // }
}

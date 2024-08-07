use zenu_cudnn_sys::{
    cudnnCreateDropoutDescriptor, cudnnDestroyDropoutDescriptor, cudnnDestroyTensorDescriptor,
    cudnnDropoutBackward, cudnnDropoutDescriptor_t, cudnnDropoutForward,
    cudnnDropoutGetReserveSpaceSize, cudnnDropoutGetStatesSize, cudnnHandle_t,
    cudnnSetDropoutDescriptor, cudnnStatus_t, cudnnTensorDescriptor_t,
};

use crate::ZENU_CUDA_STATE;

use super::{error::ZenuCudnnError, tensor_descriptor_2d, tensor_descriptor_4d};

fn dropout_descriptor() -> Result<cudnnDropoutDescriptor_t, ZenuCudnnError> {
    let mut dropout: cudnnDropoutDescriptor_t = std::ptr::null_mut();
    unsafe {
        let status = cudnnCreateDropoutDescriptor(&mut dropout as *mut cudnnDropoutDescriptor_t);
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
            x as *const std::ffi::c_void,
            tensor_desc,
            y as *mut std::ffi::c_void,
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
            dy as *const std::ffi::c_void,
            tensor_desc,
            dx as *mut std::ffi::c_void,
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
    Ok(size_in_bytes as i32)
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
        let shape = shape.into_iter().map(|x| *x as i32).collect::<Vec<i32>>();
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

    pub fn get_reserve_space_size(&self) -> usize {
        self.reserve_space_size
    }

    pub fn get_state_size(&self) -> i32 {
        self.state_size
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
            self.get_state_size() as usize,
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
mod dropout {
    use crate::runtime::{cuda_copy, cuda_malloc, cuda_malloc_bytes, ZenuCudaMemCopyKind};

    use super::*;

    #[test]
    fn test_dropout() {
        let dropout = DropoutConfig::<f32>::new(&[2, 3, 2, 2]).unwrap();
        let space_size = dropout.get_reserve_space_size();
        let state_size = dropout.get_state_size();
        let space = cuda_malloc_bytes(space_size).unwrap();
        let state = cuda_malloc::<u8>(state_size as usize).unwrap();
        dropout.set(0.5, 0, state as *mut ::libc::c_void).unwrap();
        let input_cpu = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ];
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
            .forward(input_gpu, output_gpu, space as *mut ::libc::c_void)
            .unwrap();
    }
}

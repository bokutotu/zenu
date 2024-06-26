#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::pub_underscore_fields)]

include!("./bindings.rs");

#[cfg(test)]
mod kernel {

    use zenu_cuda_runtime_sys::{cudaMalloc, cudaMemcpy, cudaMemcpyKind};

    use crate::array_scalar_add_float;

    #[test]
    fn test_add() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        // let b: Vec<f32> = vec![0., 0., 0.];

        let a_gpu: *mut f32 = std::ptr::null_mut();
        let a_gpu_gpu = &a_gpu as *const *mut f32 as *mut *mut libc::c_void;
        unsafe { cudaMalloc(a_gpu_gpu, 3 * std::mem::size_of::<f32>()) };
        let a_gpu = unsafe { *a_gpu_gpu };

        let b_gpu: *mut f32 = std::ptr::null_mut();
        let b_gpu_gpu = &b_gpu as *const *mut f32 as *mut *mut libc::c_void;
        unsafe { cudaMalloc(b_gpu_gpu, 3 * std::mem::size_of::<f32>()) };
        let b_gpu = unsafe { *b_gpu_gpu };

        unsafe {
            cudaMemcpy(
                a_gpu,
                a.as_ptr() as *const libc::c_void,
                3 * std::mem::size_of::<f32>(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        };

        unsafe { array_scalar_add_float(a_gpu as *mut f32, 3, 1, 1., b_gpu as *mut f32, 1) };
    }
}

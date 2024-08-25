#[cfg(feature = "nvidia")]
use std::any::TypeId;

use crate::{
    device::{cpu::Cpu, Device},
    dim::{default_stride, DimTrait},
    index::Index0D,
    matrix::{Matrix, Ref, Repr},
    num::Num,
};

pub trait ReluOps {
    fn relu<T: Num>(
        input: *const T,
        output: *mut T,
        alpha: T,
        size: usize,
        input_stride: usize,
        output_stride: usize,
    );
    fn relu_backward_mask<T: Num>(
        input: *const T,
        mask: *mut T,
        alpha: T,
        size: usize,
        input_stride: usize,
        mask_stride: usize,
    );
}

impl ReluOps for Cpu {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn relu<T: Num>(
        input: *const T,
        output: *mut T,
        alpha: T,
        size: usize,
        input_stride: usize,
        output_stride: usize,
    ) {
        unsafe {
            if input_stride == 1 && output_stride == 1 {
                for i in 0..size {
                    *output.add(i) = if *input.add(i) > T::zero() {
                        *input.add(i)
                    } else {
                        alpha * *input.add(i)
                    };
                }
            } else {
                for i in 0..size {
                    *output.add(i * output_stride) = if *input.add(i * input_stride) > T::zero() {
                        *input.add(i * input_stride)
                    } else {
                        alpha * *input.add(i * input_stride)
                    };
                }
            }
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn relu_backward_mask<T: Num>(
        input: *const T,
        mask: *mut T,
        alpha: T,
        size: usize,
        input_stride: usize,
        mask_stride: usize,
    ) {
        unsafe {
            if input_stride == 1 && mask_stride == 1 {
                for i in 0..size {
                    *mask.add(i) = if *input.add(i) > T::zero() {
                        T::one()
                    } else {
                        alpha * T::minus_one()
                    };
                }
            } else {
                for i in 0..size {
                    *mask.add(i * mask_stride) = if *input.add(i * input_stride) > T::zero() {
                        T::one()
                    } else {
                        alpha * T::minus_one()
                    };
                }
            }
        }
    }
}

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

#[cfg(feature = "nvidia")]
use zenu_cuda::kernel::activation::{relu, relu_backward_mask};

#[cfg(feature = "nvidia")]
impl ReluOps for Nvidia {
    fn relu<T: Num>(
        input: *const T,
        output: *mut T,
        alpha: T,
        size: usize,
        input_stride: usize,
        output_stride: usize,
    ) {
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let alpha: f32 = unsafe { *(&alpha as *const T as *const f32) };
            relu(
                input as *mut f32,
                output as *mut f32,
                alpha,
                size,
                input_stride,
                output_stride,
            )
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let alpha: f64 = unsafe { *(&alpha as *const T as *const f64) };
            relu(
                input as *mut f64,
                output as *mut f64,
                alpha,
                size,
                input_stride,
                output_stride,
            )
        } else {
            panic!("Unsupported data type");
        }
    }

    fn relu_backward_mask<T: Num>(
        input: *const T,
        mask: *mut T,
        alpha: T,
        size: usize,
        input_stride: usize,
        mask_stride: usize,
    ) {
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let alpha: f32 = unsafe { *(&alpha as *const T as *const f32) };
            relu_backward_mask(
                input as *mut f32,
                mask as *mut f32,
                alpha,
                size,
                input_stride,
                mask_stride,
            )
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let alpha: f64 = unsafe { *(&alpha as *const T as *const f64) };
            relu_backward_mask(
                input as *mut f64,
                mask as *mut f64,
                alpha,
                size,
                input_stride,
                mask_stride,
            )
        } else {
            panic!("Unsupported data type");
        }
    }
}

impl<T: Num, S: DimTrait, D: Device> Matrix<Ref<&mut T>, S, D> {
    pub fn relu<R: Repr<Item = T>, SO: DimTrait>(&self, other: &Matrix<R, SO, D>, alpha: T) {
        if self.shape().slice() != other.shape().slice() {
            panic!("shape mismatch");
        }

        let len = self.shape().len();
        let is_self_default_stride = self.shape_stride().is_default_stride();
        let is_other_default_stride = other.shape_stride().is_default_stride();

        if len == 0 {
            D::relu(other.as_ptr(), self.as_mut_ptr(), alpha, 1, 1, 1);
        } else if is_self_default_stride && is_other_default_stride {
            let num_elm = self.shape().num_elm();
            let stride_self = self.stride()[len - 1];
            let stride_other = other.stride()[len - 1];
            let self_ptr = self.as_mut_ptr();
            let other_ptr = other.as_ptr();
            D::relu(
                other_ptr,
                self_ptr,
                alpha,
                num_elm,
                stride_other,
                stride_self,
            );
        } else {
            for i in 0..self.shape()[0] {
                self.index_axis_mut_dyn(Index0D::new(i))
                    .relu(&other.index_axis_dyn(Index0D::new(i)), alpha);
            }
        }
    }

    pub fn relu_backward_mask<R: Repr<Item = T>, SO: DimTrait>(
        &self,
        other: &Matrix<R, SO, D>,
        alpha: T,
    ) {
        if self.shape().slice() != other.shape().slice() {
            panic!("shape mismatch");
        }

        let len = self.shape().len();
        let is_self_default_stride = self.stride() == default_stride(self.shape());
        let is_other_default_stride = other.stride() == default_stride(other.shape());

        if len == 0 {
            D::relu_backward_mask(other.as_ptr(), self.as_mut_ptr(), alpha, 1, 1, 1);
        } else if is_self_default_stride && is_other_default_stride {
            let num_elm = self.shape().num_elm();
            let stride_self = self.stride()[len - 1];
            let stride_other = other.stride()[len - 1];
            let self_ptr = self.as_mut_ptr();
            let other_ptr = other.as_ptr();
            D::relu_backward_mask(
                other_ptr,
                self_ptr,
                alpha,
                num_elm,
                stride_other,
                stride_self,
            );
        } else {
            for i in 0..self.shape()[0] {
                self.index_axis_mut_dyn(Index0D::new(i))
                    .relu_backward_mask(&other.index_axis_dyn(Index0D::new(i)), alpha);
            }
        }
    }
}

#[cfg(test)]
mod relu {
    use zenu_test::{assert_mat_eq_epsilon, run_mat_test};

    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    fn relu<D: Device>() {
        let x = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0, -1.0, 0.0, 2.0], [2, 2]);
        let mut y = Matrix::<Owned<f32>, DimDyn, D>::zeros([2, 2]);
        y.to_ref_mut().relu(&x.to_ref(), 0.0);
        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0, 0.0, 0.0, 2.0], [2, 2]);
        assert_mat_eq_epsilon!(y.to_ref(), ans.to_ref(), 1.0e-6);
    }
    run_mat_test!(relu, relu_cpu, relu_nvidia);

    fn relu_backward_mask<D: Device>() {
        let x = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0, -1.0, 0.0, 2.0], [2, 2]);
        let mut y = Matrix::<Owned<f32>, DimDyn, D>::zeros([2, 2]);
        y.to_ref_mut().relu_backward_mask(&x.to_ref(), 0.0);
        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0, 0.0, 0.0, 1.0], [2, 2]);
        assert_mat_eq_epsilon!(y.to_ref(), ans.to_ref(), 1.0e-6);
    }
    run_mat_test!(
        relu_backward_mask,
        relu_backward_mask_cpu,
        relu_backward_mask_nvidia
    );

    fn relu_0d<D: Device>() {
        let x = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0], []);
        let mut y = Matrix::<Owned<f32>, DimDyn, D>::zeros([]);
        y.to_ref_mut().relu(&x.to_ref(), 0.0);
        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0], []);
        assert_mat_eq_epsilon!(y.to_ref(), ans.to_ref(), 1.0e-6);
    }
    run_mat_test!(relu_0d, relu_0d_cpu, relu_0d_nvidia);

    fn relu_backward_mask_0d<D: Device>() {
        let x = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0], []);
        let mut y = Matrix::<Owned<f32>, DimDyn, D>::zeros([]);
        y.to_ref_mut().relu_backward_mask(&x.to_ref(), 0.0);
        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0], []);
        assert_mat_eq_epsilon!(y.to_ref(), ans.to_ref(), 1.0e-6);
    }
    run_mat_test!(
        relu_backward_mask_0d,
        relu_backward_mask_0d_cpu,
        relu_backward_mask_0d_nvidia
    );
}

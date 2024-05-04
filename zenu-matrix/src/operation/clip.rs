use crate::{
    device::{cpu::Cpu, DeviceBase},
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{Matrix, Owned, Ref, Repr},
    num::Num,
};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

#[cfg(feature = "nvidia")]
use zenu_cuda::kernel::*;

pub trait ClipOps {
    fn clip<T: Num>(
        input: *const T,
        output: *mut T,
        size: usize,
        stride_in: usize,
        stride_out: usize,
        min: T,
        max: T,
    );
    fn clip_assign<T: Num>(input: *mut T, size: usize, stride: usize, min: T, max: T);
}

impl ClipOps for Cpu {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn clip<T: Num>(
        input: *const T,
        output: *mut T,
        size: usize,
        stride_in: usize,
        stride_out: usize,
        min: T,
        max: T,
    ) {
        let input = unsafe { std::slice::from_raw_parts(input, size * stride_in) };
        let output = unsafe { std::slice::from_raw_parts_mut(output, size * stride_out) };
        for i in 0..size {
            let mut x = input[i * stride_in];
            if x < min {
                x = min;
            } else if x > max {
                x = max;
            }
            output[i * stride_out] = x;
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn clip_assign<T: Num>(input: *mut T, size: usize, stride: usize, min: T, max: T) {
        let input = unsafe { std::slice::from_raw_parts_mut(input, size * stride) };
        for i in 0..size {
            let mut x = input[i * stride];
            if x < min {
                x = min;
            } else if x > max {
                x = max;
            }
            input[i * stride] = x;
        }
    }
}

#[cfg(feature = "nvidia")]
impl ClipOps for Nvidia {
    fn clip<T: Num>(
        input: *const T,
        output: *mut T,
        size: usize,
        stride_in: usize,
        stride_out: usize,
        min: T,
        max: T,
    ) {
        clip(input, output, size, stride_in, stride_out, min, max)
    }

    fn clip_assign<T: Num>(input: *mut T, size: usize, stride: usize, min: T, max: T) {
        clip_assign(input, size, stride, min, max)
    }
}

fn clip_1d<T: Num, R: Repr<Item = T>, SI: DimTrait, SO: DimTrait, D: DeviceBase + ClipOps>(
    input: &Matrix<R, SI, D>,
    output: &Matrix<Ref<&mut T>, SO, D>,
    min: T,
    max: T,
) {
    let size = input.shape()[0];
    let stride_in = input.stride()[0];
    let stride_out = output.stride()[0];
    let input_ptr = input.as_ptr();
    let output_ptr = output.as_mut_ptr();
    D::clip(input_ptr, output_ptr, size, stride_in, stride_out, min, max);
}

fn clip_assign_1d<T: Num, S: DimTrait, D: DeviceBase + ClipOps>(
    input: &Matrix<Ref<&mut T>, S, D>,
    min: T,
    max: T,
) {
    let size = input.shape()[0];
    let stride = input.stride()[0];
    let input_ptr = input.as_mut_ptr();
    D::clip_assign(input_ptr, size, stride, min, max);
}

fn clip_inner<T: Num, D: DeviceBase + ClipOps>(
    input: &Matrix<Ref<&T>, DimDyn, D>,
    output: &Matrix<Ref<&mut T>, DimDyn, D>,
    min: T,
    max: T,
) {
    if input.shape().len() == 1 {
        clip_1d(input, output, min, max);
    } else if input.shape().is_empty() {
        unimplemented!();
    } else {
        for i in 0..(input.shape()[0]) {
            clip_inner(
                &input.index_axis_dyn(Index0D::new(i)),
                &output.index_axis_mut_dyn(Index0D::new(i)),
                min,
                max,
            );
        }
    }
}

fn clip_assign_inner<T: Num, D: DeviceBase + ClipOps>(
    input: &Matrix<Ref<&mut T>, DimDyn, D>,
    min: T,
    max: T,
) {
    if input.shape().len() == 1 {
        clip_assign_1d(input, min, max);
    } else if input.shape().is_empty() {
        unimplemented!();
    } else {
        for i in 0..(input.shape()[0]) {
            clip_assign_inner(&input.index_axis_mut(Index0D::new(i)), min, max);
        }
    }
}

impl<R: Repr, S: DimTrait, D: DeviceBase + ClipOps> Matrix<R, S, D> {
    pub fn clip(&self, min: R::Item, max: R::Item) -> Matrix<Owned<R::Item>, S, D> {
        let mut output = Matrix::<_, S, D>::zeros_like(self);
        let s_v = self.to_ref().into_dyn_dim();

        clip_inner(&s_v, &output.to_ref_mut().into_dyn_dim(), min, max);

        output
    }
}

impl<T: Num, D: DeviceBase + ClipOps> Matrix<Ref<&mut T>, DimDyn, D> {
    pub fn clip_assign(&self, min: T, max: T) {
        clip_assign_inner(self, min, max);
    }
}

#[cfg(test)]
mod clip {
    use crate::{
        device::DeviceBase,
        dim::DimDyn,
        matrix::{Matrix, Owned},
        operation::{asum::Asum, basic_operations::SubOps},
    };

    use super::ClipOps;

    fn clip_1d<D: DeviceBase + ClipOps + SubOps + Asum>() {
        let mut a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], [4]);
        let b = a.clip(2.0, 3.0);
        let ans: Matrix<_, DimDyn, _> = Matrix::from_vec(vec![2.0, 2.0, 3.0, 3.0], [4]);
        let diff = b - ans.to_ref();
        let diff_asum = diff.asum();
        assert_eq!(diff_asum, 0.0);

        a.to_ref_mut().clip_assign(2.0, 3.0);
        let diff = a - ans.to_ref();
        let diff_asum = diff.asum();
        assert_eq!(diff_asum, 0.0);
    }
    #[test]
    fn clip_1d_cpu() {
        clip_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn clip_1d_nvidia() {
        clip_1d::<crate::device::nvidia::Nvidia>();
    }

    fn clip_2d<D: DeviceBase + ClipOps + SubOps + Asum>() {
        let mut a: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b = a.clip(2.0, 3.0);
        let ans: Matrix<_, DimDyn, _> = Matrix::from_vec(vec![2.0, 2.0, 3.0, 3.0], [2, 2]);
        let diff = b - ans;
        let diff_asum = diff.asum();
        assert_eq!(diff_asum, 0.0);

        a.to_ref_mut().clip_assign(2.0, 3.0);
        let ans: Matrix<_, DimDyn, _> = Matrix::from_vec(vec![2.0, 2.0, 3.0, 3.0], [2, 2]);
        let diff = a - ans;
        let diff_asum = diff.asum();
        assert_eq!(diff_asum, 0.0);
    }
    #[test]
    fn clip_2d_cpu() {
        clip_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn clip_2d_nvidia() {
        println!("here");
        clip_2d::<crate::device::nvidia::Nvidia>();
    }
}

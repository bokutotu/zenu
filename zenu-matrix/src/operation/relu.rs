use crate::{
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{AsMutPtr, AsPtr, IndexAxisDyn, IndexAxisMutDyn, MatrixBase, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
};

pub trait Relu<T: Num> {
    fn relu(&mut self, source: Matrix<ViewMem<T>, DimDyn>);
    fn relu_backward_mask(&mut self, source: Matrix<ViewMem<T>, DimDyn>);
}

impl<'a, T: Num> Relu<T> for Matrix<ViewMutMem<'a, T>, DimDyn> {
    fn relu(&mut self, source: Matrix<ViewMem<T>, DimDyn>) {
        if self.shape() != source.shape() {
            panic!("shape mismatch");
        }

        let len = self.shape().len();
        if len == 0 {
            unsafe {
                *self.as_mut_ptr() = if *source.as_ptr() > T::zero() {
                    *source.as_ptr()
                } else {
                    T::zero()
                };
            };
        } else if len == 1 {
            let num_elm = self.shape().num_elm();
            let stride_self = self.stride()[0];
            let stride_source = source.stride()[0];
            let self_ptr = self.as_mut_slice();
            let source_ptr = source.as_slice();
            relu_kernel_cpu(source_ptr, self_ptr, num_elm, stride_source, stride_self);
        } else {
            for i in 0..self.shape()[0] {
                self.to_view_mut()
                    .index_axis_mut_dyn(Index0D::new(i))
                    .relu(source.index_axis_dyn(Index0D::new(i)).clone());
            }
        }
    }

    fn relu_backward_mask(&mut self, source: Matrix<ViewMem<T>, DimDyn>) {
        if self.shape() != source.shape() {
            panic!("shape mismatch");
        }

        let len = self.shape().len();
        if len == 0 {
            unsafe {
                *self.as_mut_ptr() = if *source.as_ptr() > T::zero() {
                    T::one()
                } else {
                    T::zero()
                };
            };
        } else if len <= 1 {
            let num_elm = self.shape().num_elm();
            let stride_self = self.stride()[0];
            let stride_source = source.stride()[0];
            let self_ptr = self.as_mut_slice();
            let source_ptr = source.as_slice();
            relu_backward_mask_kernel_cpu(
                source_ptr,
                self_ptr,
                num_elm,
                stride_source,
                stride_self,
            );
        } else {
            for i in 0..self.shape()[0] {
                self.to_view_mut()
                    .index_axis_mut_dyn(Index0D::new(i))
                    .relu_backward_mask(source.index_axis_dyn(Index0D::new(i)).clone());
            }
        }
    }
}

fn relu_kernel_cpu<T: Num>(x: &[T], y: &mut [T], len: usize, incx: usize, incy: usize) {
    if incx == 1 && incy == 1 {
        for i in 0..len {
            y[i] = if x[i] > T::zero() { x[i] } else { T::zero() };
        }
    } else {
        for i in 0..len {
            y[i * incy] = if x[i * incx] > T::zero() {
                x[i * incx]
            } else {
                T::zero()
            };
        }
    }
}

fn relu_backward_mask_kernel_cpu<T: Num>(
    x: &[T],
    mask: &mut [T],
    len: usize,
    incx: usize,
    incmask: usize,
) {
    if incx == 1 && incmask == 1 {
        for i in 0..len {
            mask[i] = if x[i] > T::zero() {
                T::one()
            } else {
                T::zero()
            };
        }
    } else {
        for i in 0..len {
            mask[i * incmask] = if x[i * incx] > T::zero() {
                T::one()
            } else {
                T::zero()
            };
        }
    }
}

#[cfg(test)]
mod relu {
    use crate::{
        matrix::{OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::{asum::Asum, zeros::Zeros},
    };

    use super::Relu;

    #[test]
    fn relu() {
        let x = OwnedMatrixDyn::from_vec(vec![1.0, -1.0, 0.0, 2.0], [2, 2]);
        let mut y = OwnedMatrixDyn::zeros([2, 2]);
        y.to_view_mut().relu(x.to_view());
        let ans = OwnedMatrixDyn::from_vec(vec![1.0, 0.0, 0.0, 2.0], [2, 2]);
        let diff = y.to_view() - ans.to_view();
        let diff_asum = diff.asum();
        assert!(diff_asum < 1.0e-6);
    }

    #[test]
    fn relu_backward_mask() {
        let x = OwnedMatrixDyn::from_vec(vec![1.0, -1.0, 0.0, 2.0], [2, 2]);
        let mut y = OwnedMatrixDyn::zeros([2, 2]);
        y.to_view_mut().relu_backward_mask(x.to_view());
        let ans = OwnedMatrixDyn::from_vec(vec![1.0, 0.0, 0.0, 1.0], [2, 2]);
        let diff = y.to_view() - ans.to_view();
        let diff_asum = diff.asum();
        assert!(diff_asum < 1.0e-6);
    }

    #[test]
    fn relu_0d() {
        let x = OwnedMatrixDyn::from_vec(vec![1.0], []);
        let mut y = OwnedMatrixDyn::zeros([]);
        y.to_view_mut().relu(x.to_view());
        let ans = OwnedMatrixDyn::from_vec(vec![1.0], []);
        let diff = y.to_view() - ans.to_view();
        let diff_asum = diff.asum();
        assert!(diff_asum < 1.0e-6);
    }

    #[test]
    fn relu_backward_mask_0d() {
        let x = OwnedMatrixDyn::from_vec(vec![1.0], []);
        let mut y = OwnedMatrixDyn::zeros([]);
        y.to_view_mut().relu_backward_mask(x.to_view());
        let ans = OwnedMatrixDyn::from_vec(vec![1.0], []);
        let diff = y.to_view() - ans.to_view();
        let diff_asum = diff.asum();
        assert!(diff_asum < 1.0e-6);
    }
}

use crate::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    matrix_iter::MatrixIter,
    memory::ToViewMutMemory,
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
};

use super::{asum::Asum, copy_from::CopyFrom, exp::Exp, max::MaxIdx};

pub trait SoftMax<T: Num> {
    fn softmax_assign(&mut self, source: Matrix<ViewMem<T>, DimDyn>, axis: usize);
}

impl<'a, T: Num, M: ToViewMutMemory<Item = T>> SoftMax<T> for Matrix<M, DimDyn> {
    fn softmax_assign(&mut self, source: Matrix<ViewMem<T>, DimDyn>, axis: usize) {
        if axis >= self.shape().len() {
            panic!("axis must be less than the number of dimensions");
        }
        self.to_view_mut().copy_from(&source);
        if self.shape().len() == 1 {
            self.to_view_mut().copy_from(&source);
            softmax_kernel_cpu(self.to_view_mut());
        } else {
            self.to_view_mut().map_axis_mut(axis, softmax_kernel_cpu);
        }
    }
}

fn softmax_kernel_cpu<T: Num>(result: Matrix<ViewMutMem<T>, DimDyn>) {
    let mut result = result;
    let max_diff = result.to_view() - result.to_view().max();
    let exp = max_diff.exp();
    let sum = exp.to_view().asum();
    let t = exp / sum;
    result.copy_from(&t.to_view());
}

#[cfg(test)]
mod softmax {
    use crate::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::{asum::Asum, zeros::Zeros},
    };

    use super::SoftMax;

    #[test]
    fn softmax_1d() {
        let a = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4.], [4]);
        let mut b = OwnedMatrixDyn::zeros([4]);
        b.softmax_assign(a.to_view(), 0);
        let ans =
            OwnedMatrixDyn::from_vec(vec![0.0320586, 0.08714432, 0.23688284, 0.64391428], [4]);
        let diff = b.to_view() - ans.to_view();
        assert!(diff.asum() < 1e-6);
    }

    #[test]
    fn softmax_2d() {
        let a = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let mut b = OwnedMatrixDyn::zeros([2, 3]);
        b.softmax_assign(a.to_view(), 1);
        let ans = OwnedMatrixDyn::from_vec(
            vec![
                0.09003057, 0.24472847, 0.66524096, 0.09003057, 0.24472847, 0.66524096,
            ],
            [2, 3],
        );
        let diff = b.to_view() - ans.to_view();
        assert!(diff.asum() < 1e-6);

        let a = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let mut b = OwnedMatrixDyn::zeros([2, 3]);
        b.softmax_assign(a.to_view(), 0);
        let ans_2 = OwnedMatrixDyn::from_vec(
            vec![
                0.04742587, 0.04742587, 0.04742587, 0.95257413, 0.95257413, 0.95257413,
            ],
            [2, 3],
        );
        println!("{:?}", b.to_view());
        println!("{:?}", ans_2.to_view());
        let diff = b.to_view() - ans_2.to_view();
        println!("{:?}", diff);
        assert!(diff.asum() < 1e-6);
    }
}

use crate::{
    device::Device,
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Ref, Repr},
    num::Num,
};

impl<T: Num, D: Device> Matrix<Ref<&mut T>, DimDyn, D> {
    pub fn softmax_assign<R: Repr<Item = T>>(&self, source: &Matrix<R, DimDyn, D>, axis: usize) {
        if axis >= self.shape().len() {
            panic!("axis must be less than the number of dimensions");
        }
        if self.shape().slice() != source.shape().slice() {
            panic!("softmax shape mismatch");
        }

        let max_diff = source.to_ref() - source.max_axis(axis, true);
        let mut output = max_diff.exp();
        let sum = output.to_ref().sum(axis, true);
        output /= sum;
        self.copy_from(&output.to_ref());
    }
}

#[cfg(test)]
mod softmax {
    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    fn softmax_1d<D: Device>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 4.], [4]);
        let mut b = Matrix::<Owned<f32>, DimDyn, D>::zeros([4]);
        b.to_ref_mut().softmax_assign(&a, 0);
        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![0.0320586, 0.08714432, 0.23688284, 0.64391428],
            [4],
        );
        let diff = b - ans;
        assert!(diff.asum() < 1e-6);
    }
    #[test]
    fn softmax_1d_cpu() {
        softmax_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn softmax_1d_cuda() {
        softmax_1d::<crate::device::nvidia::Nvidia>();
    }

    fn softmax_2d<D: Device>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let mut b = Matrix::<Owned<f32>, DimDyn, D>::zeros([2, 3]);
        b.to_ref_mut().softmax_assign(&a, 1);
        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![
                0.09003057, 0.24472847, 0.66524096, 0.09003057, 0.24472847, 0.66524096,
            ],
            [2, 3],
        );
        let diff = b - ans;
        assert!(diff.asum() < 1e-6);

        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let mut b = Matrix::<Owned<f32>, DimDyn, D>::zeros([2, 3]);
        b.to_ref_mut().softmax_assign(&a, 0);
        let ans_2 = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![
                0.04742587, 0.04742587, 0.04742587, 0.95257413, 0.95257413, 0.95257413,
            ],
            [2, 3],
        );
        let diff = b - ans_2;
        assert!(diff.asum() < 1e-6);
    }
    #[test]
    fn softmax_2d_cpu() {
        softmax_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn softmax_2d_cuda() {
        softmax_2d::<crate::device::nvidia::Nvidia>();
    }
}

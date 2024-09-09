use crate::{
    device::Device,
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{Matrix, Ref, Repr},
    num::Num,
};

impl<T: Num, D: Device> Matrix<Ref<&mut T>, DimDyn, D> {
    #[expect(clippy::missing_panics_doc)]
    pub fn broadcast<R: Repr<Item = T>>(&self, source: &Matrix<R, DimDyn, D>) {
        let source = source.to_ref();
        if !(self.shape().is_include(source.shape())
            || self.shape().is_include_bradcast(source.shape()))
        {
            panic!("!self.shape().is_include(source.shape())");
        }
        if self.shape() == source.shape() {
            self.copy_from(&source);
            return;
        }
        if !source.shape().is_empty() && source.shape()[0] == 1 {
            let source = source.index_axis_dyn(Index0D::new(0));
            self.broadcast(&source);
            return;
        }

        let diff_len = self.shape().len() - source.shape().len();

        if diff_len == 1 {
            for i in 0..self.shape()[0] {
                let to = self.index_axis_mut_dyn(Index0D::new(i));
                to.copy_from(&source);
            }
        } else {
            for i in 0..self.shape()[0] {
                let to = self.index_axis_mut_dyn(Index0D::new(i));
                to.broadcast(&source);
            }
        }
    }
}

#[cfg(test)]
mod broadcast_test {
    #![expect(clippy::float_cmp)]
    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    fn broadcast_1d_0d<D: Device>() {
        let source: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1.], []);
        let mut res: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([3]);
        res.to_ref_mut().broadcast(&source.to_ref());
        let ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1., 1., 1.], [3]);
        let diff = ans.to_ref() - res.to_ref();
        let diff_sum = diff.to_ref().asum();
        assert_eq!(diff_sum, 0.);
    }
    #[test]
    fn broadcast_1d_0d_cpu() {
        broadcast_1d_0d::<crate::device::cpu::Cpu>();
    }
    #[test]
    #[cfg(feature = "nvidia")]
    fn broadcast_1d_0d_nvidia() {
        broadcast_1d_0d::<crate::device::nvidia::Nvidia>();
    }

    // #[test]
    fn broadcast_2d_0d<D: Device>() {
        let source: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1.], []);
        let mut res: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 3]);
        res.to_ref_mut().broadcast(&source.to_ref());
        let ans: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 1., 1., 1., 1., 1.], [2, 3]);
        let diff = ans.to_ref() - res.to_ref();
        let diff_sum = diff.to_ref().asum();
        assert_eq!(diff_sum, 0.);
    }
    #[test]
    fn broadcast_2d_0d_cpu() {
        broadcast_2d_0d::<crate::device::cpu::Cpu>();
    }
    #[test]
    #[cfg(feature = "nvidia")]
    fn broadcast_2d_0d_nvidia() {
        broadcast_2d_0d::<crate::device::nvidia::Nvidia>();
    }

    // #[test]
    fn broadcast_2d_1d<D: Device>() {
        let source: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1., 2., 3.], [3]);
        let mut res: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 3]);
        res.to_ref_mut().broadcast(&source.to_ref());
        let ans: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 2., 3., 1., 2., 3.], [2, 3]);
        let diff = ans.to_ref() - res.to_ref();
        let diff_sum = diff.to_ref().asum();
        assert_eq!(diff_sum, 0.);
    }
    #[test]
    fn broadcast_2d_1d_cpu() {
        broadcast_2d_1d::<crate::device::cpu::Cpu>();
    }
    #[test]
    #[cfg(feature = "nvidia")]
    fn broadcast_2d_1d_nvidia() {
        broadcast_2d_1d::<crate::device::nvidia::Nvidia>();
    }

    fn broadcast_4d_2d<D: Device>() {
        let source: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1., 2.], [1, 2]);
        let mut res: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 3, 1, 2]);
        res.to_ref_mut().broadcast(&source.to_ref());
        let ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.],
            [2, 3, 1, 2],
        );
        let diff = ans.to_ref() - res.to_ref();
        let diff_sum = diff.to_ref().asum();
        assert_eq!(diff_sum, 0.);
    }
    #[test]
    fn broadcast_4d_2d_cpu() {
        broadcast_4d_2d::<crate::device::cpu::Cpu>();
    }
    #[test]
    #[cfg(feature = "nvidia")]
    fn broadcast_4d_2d_nvidia() {
        broadcast_4d_2d::<crate::device::nvidia::Nvidia>();
    }

    // #[test]
    fn broadcast_4d_4d<D: Device>() {
        let source: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 2., 3., 4.], [1, 1, 1, 4]);
        let mut res: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 3, 4, 4]);
        res.to_ref_mut().broadcast(&source.to_ref());
        let ans: Matrix<_, DimDyn, D> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1.,
                2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2.,
                3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3.,
                4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
                1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
            ],
            [2, 3, 4, 4],
        );

        let diff = ans - res;
        let diff_sum = diff.asum();
        assert_eq!(diff_sum, 0.);
    }
    #[test]
    fn broadcast_4d_4d_cpu() {
        broadcast_4d_4d::<crate::device::cpu::Cpu>();
    }
    #[test]
    #[cfg(feature = "nvidia")]
    fn broadcast_4d_4d_nvidia() {
        broadcast_4d_4d::<crate::device::nvidia::Nvidia>();
    }
}

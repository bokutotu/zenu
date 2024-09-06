use crate::{
    device::Device,
    dim::DimDyn,
    matrix::{Matrix, Repr},
};

impl<R: Repr, D: Device> Matrix<R, DimDyn, D> {
    pub fn add_axis(&mut self, axis: usize) {
        let shape_stride = self.shape_stride();
        let shape_stride = shape_stride.add_axis(axis);
        self.update_shape(shape_stride.shape());
        self.update_stride(shape_stride.stride());
    }
}

#[cfg(test)]
mod add_axis_test {
    #![allow(clippy::float_cmp)]
    use crate::{
        device::Device,
        dim::{DimDyn, DimTrait},
        matrix::{Matrix, Owned},
    };

    fn test<D: Device>() {
        let mut a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1., 2., 3., 4.], [2, 2]);
        a.add_axis(0);
        assert_eq!(a.shape().slice(), [1, 2, 2]);
        let ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1., 2., 3., 4.], [1, 2, 2]);
        let diff = a.to_ref() - ans.to_ref();
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }
    #[test]
    fn cpu() {
        test::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn nvidia() {
        test::<crate::device::nvidia::Nvidia>();
    }
}

use crate::{dim::DimDyn, matrix::MatrixBase, matrix_impl::Matrix, memory::Memory};

pub trait MatrixAddAxis {
    fn add_axis(&mut self, axis: usize);
}

impl<M: Memory> MatrixAddAxis for Matrix<M, DimDyn> {
    fn add_axis(&mut self, axis: usize) {
        let shape_stride = self.shape_stride();
        let shape_stride = shape_stride.add_axis(axis);
        self.update_shape(shape_stride.shape());
        self.update_stride(shape_stride.stride());
    }
}

#[cfg(test)]
mod add_axis {
    use crate::{
        dim::DimTrait,
        matrix::{MatrixBase, OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::{add_axis::MatrixAddAxis, asum::Asum},
    };

    #[test]
    fn test() {
        let mut a = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4.], &[2, 2]);
        a.add_axis(0);
        assert_eq!(a.shape().slice(), [1, 2, 2]);
        let ans = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4.], &[1, 2, 2]);
        let diff = a.to_view() - ans.to_view();
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }
}

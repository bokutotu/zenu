use crate::{
    dim::{default_stride, DimDyn, DimTrait},
    matrix::{MatrixBase, OwnedMatrix, ToViewMatrix},
    matrix_impl::{Matrix, OwnedMatrixDyn},
    memory::ToViewMemory,
    memory_impl::{OwnedMem, ViewMem},
    num::Num,
    shape_stride::ShapeStride,
};

use super::to_default_stride::ToDefaultStride;

pub trait Reshape<T: Num>: ToViewMatrix {
    fn reshape<I: Into<DimDyn>>(&self, new_shape: I) -> Matrix<ViewMem<T>, DimDyn>;
    fn reshape_new_matrix<I: Into<DimDyn>>(&self, new_shape: I) -> Matrix<OwnedMem<T>, DimDyn>;
}

impl<T: Num, D: DimTrait, V: ToViewMemory<Item = T>> Reshape<T> for Matrix<V, D> {
    fn reshape<I: Into<DimDyn>>(&self, new_shape: I) -> Matrix<ViewMem<T>, DimDyn> {
        let new_shape = new_shape.into();
        assert_eq!(
            self.shape().num_elm(),
            new_shape.num_elm(),
            "Number of elements must be the same"
        );
        assert!(
            self.shape_stride().is_default_stride(),
            r#"""
`reshape` method is not alloc new memory. 
So, This matrix is not default stride, it is not allowed to use `reshape` method. 
Use `reshape_new_matrix` method instead.
            """#
        );
        let new_stride = default_stride(new_shape);
        let mut result = self.to_view().into_dyn_dim();
        result.update_shape_stride(ShapeStride::new(new_shape, new_stride));
        result
    }

    fn reshape_new_matrix<I: Into<DimDyn>>(&self, new_shape: I) -> Matrix<OwnedMem<T>, DimDyn> {
        let new_shape = new_shape.into();
        assert_eq!(
            self.shape().num_elm(),
            new_shape.num_elm(),
            "Number of elements must be the same"
        );
        let new_stride = default_stride(new_shape);

        let mut default_stride_matrix = self.to_view().to_default_stride();
        default_stride_matrix.update_shape_stride(ShapeStride::new(new_shape, new_stride));
        default_stride_matrix
    }
}

#[cfg(test)]
mod reshape {
    use crate::{
        dim::DimTrait,
        matrix::{MatrixBase, OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::{asum::Asum, transpose::TransposeInplace},
    };

    use super::Reshape;

    #[test]
    fn reshape_3d_1d() {
        let a = OwnedMatrixDyn::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            ],
            [2, 3, 3],
        );
        let b = a.reshape([18]);
        let ans = OwnedMatrixDyn::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            ],
            [18],
        );
        assert_eq!(b.shape().slice(), ans.shape().slice());
        assert!((b - ans).to_view().asum() < 1e-6);
    }

    #[test]
    fn reshape_1d_3d() {
        let a = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6.], [6]);
        let b = a.reshape([2, 3, 1]);
        let ans = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3, 1]);
        assert_eq!(b.shape().slice(), ans.shape().slice());
        assert!((b - ans).to_view().asum() < 1e-6);
    }

    #[test]
    fn reshape_new_matrix_3d_1d() {
        let a = OwnedMatrixDyn::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            ],
            [2, 3, 3],
        );
        let a = a.transepose_by_index(&[2, 1, 0]);
        let b = a.reshape_new_matrix([18]);
        println!("{:?}", b);
        let ans = OwnedMatrixDyn::from_vec(
            vec![
                1., 10., 4., 13., 7., 16., 2., 11., 5., 14., 8., 17., 3., 12., 6., 15., 9., 18.,
            ],
            [18],
        );
        assert_eq!(b.shape().slice(), ans.shape().slice());
        assert!((b - ans).to_view().asum() < 1e-6);
    }
}

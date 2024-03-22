use crate::{
    constructor::zeros::Zeros,
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{IndexAxisMutDyn, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory_impl::OwnedMem,
    num::Num,
    operation::copy_from::CopyFrom,
};

pub fn concat<T: Num, M: ToViewMatrix<Item = T>>(matrix: &[M]) -> Matrix<OwnedMem<T>, DimDyn> {
    let first_shape = matrix[0].shape();
    for m in matrix.iter().skip(1) {
        if m.shape() != first_shape {
            panic!("All matrices must have the same shape");
        }
    }
    if first_shape.len() == 4 {
        panic!("Concatenation of 4D matrices is not supported");
    }

    let mut shape = DimDyn::default();
    shape.push_dim(matrix.len());
    for d in first_shape {
        shape.push_dim(d);
    }

    let mut result = Matrix::zeros(shape);

    for (i, m) in matrix.iter().enumerate() {
        let view = m.to_view().into_dyn_dim();
        result
            .to_view_mut()
            .index_axis_mut_dyn(Index0D::new(i))
            .copy_from(&view);
    }

    result
}

#[cfg(test)]
mod concat {
    use crate::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    #[test]
    fn cat_1d() {
        let a = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], [3]);
        let b = OwnedMatrixDyn::from_vec(vec![4., 5., 6.], [3]);
        let c = OwnedMatrixDyn::from_vec(vec![7., 8., 9.], [3]);
        let result = super::concat(&[a, b, c]);

        let ans = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], [3, 3]);

        let diff = result - ans;
        assert_eq!(diff.asum(), 0.);
    }

    #[test]
    fn cal_2d() {
        let a = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4.], [2, 2]);
        let b = OwnedMatrixDyn::from_vec(vec![5., 6., 7., 8.], [2, 2]);
        let c = OwnedMatrixDyn::from_vec(vec![9., 10., 11., 12.], [2, 2]);
        let result = super::concat(&[a, b, c]);

        let ans = OwnedMatrixDyn::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [3, 2, 2],
        );

        let diff = result - ans;
        assert_eq!(diff.asum(), 0.);
    }
}

use crate::{
    blas::{Blas, BlasLayout, BlasTrans},
    dim::Dim2,
    matrix::{MatrixBase, ViewMatrix, ViewMutMatix},
    num::Num,
};

fn get_leading_dim(shape: Dim2, is_transpose: bool) -> usize {
    if is_transpose {
        shape[0]
    } else {
        shape[1]
    }
}

pub(crate) fn gemm_shape_check<T, A, B, C>(a: &A, b: &B, c: &C)
where
    T: Num,
    A: ViewMatrix + MatrixBase<Dim = Dim2, Item = T>,
    B: ViewMatrix + MatrixBase<Dim = Dim2, Item = T>,
    C: ViewMutMatix + MatrixBase<Dim = Dim2, Item = T>,
{
    let c_shape = c.shape();
    let a_shape = a.shape();
    let b_shape = b.shape();

    let is_transposed_c = c.shape_stride().is_transposed();

    if is_transposed_c {
        panic!("The output matrix C must not be transposed.");
    }

    if a_shape[0] != c_shape[0] {
        panic!("The number of rows of matrix A must match the number of rows of matrix C.");
    }

    if b_shape[1] != c_shape[1] {
        panic!("The number of columns of matrix B must match the number of columns of matrix C.");
    }

    if a_shape[1] != b_shape[0] {
        panic!("The number of columns of matrix A must match the number of rows of matrix B.");
    }

    if a_shape[0] == 0
        || a_shape[1] == 0
        || b_shape[0] == 0
        || b_shape[1] == 0
        || c_shape[0] == 0
        || c_shape[1] == 0
    {
        panic!("The dimensions of the input and output matrices must be greater than 0.");
    }
}

pub(crate) fn gemm_unchecked<T, A, B, C>(a: A, b: B, mut c: C, alpha: T, beta: T)
where
    T: Num,
    A: ViewMatrix + MatrixBase<Dim = Dim2, Item = T>,
    B: ViewMatrix + MatrixBase<Dim = Dim2, Item = T>,
    C: ViewMutMatix + MatrixBase<Dim = Dim2, Item = T>,
{
    let c_shape = c.shape();
    let a_shape = a.shape();
    let b_shape = b.shape();

    let is_transposed_a = a.shape_stride().is_transposed();
    let is_transposed_b = b.shape_stride().is_transposed();
    let is_transposed_c = c.shape_stride().is_transposed();

    let m = a_shape[0];
    let n = b_shape[1];
    let k = a_shape[1];

    let leading_dim_a = get_leading_dim(a_shape, is_transposed_a);
    let leading_dim_b = get_leading_dim(b_shape, is_transposed_b);
    let leading_dim_c = get_leading_dim(c_shape, is_transposed_c);

    let get_trans = |is_trans| {
        if is_trans {
            BlasTrans::Ordinary
        } else {
            BlasTrans::None
        }
    };

    let transa = get_trans(is_transposed_a);
    let transb = get_trans(is_transposed_b);

    A::Blas::gemm(
        BlasLayout::RowMajor,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a.as_ptr(),
        leading_dim_a,
        b.as_ptr(),
        leading_dim_b,
        beta,
        c.as_mut_ptr(),
        leading_dim_c,
    );
}

pub fn gemm<T, A, B, C>(a: A, b: B, c: C, alpha: T, beta: T)
where
    T: Num,
    A: ViewMatrix + MatrixBase<Dim = Dim2, Item = T>,
    B: ViewMatrix + MatrixBase<Dim = Dim2, Item = T>,
    C: ViewMutMatix + MatrixBase<Dim = Dim2, Item = T>,
{
    gemm_shape_check(&a, &b, &c);
    gemm_unchecked(a, b, c, alpha, beta);
}
#[cfg(test)]
mod gemm {
    use crate::{
        matrix::{IndexItem, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::OwnedMatrix2D,
        operation::transpose::Transpose,
    };

    use super::gemm;

    #[test]
    fn non_transposed() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = vec![0.0, 0.0, 0.0, 0.0];

        let a = OwnedMatrix2D::from_vec(a, [2, 2]);
        let b = OwnedMatrix2D::from_vec(b, [2, 2]);
        let mut c = OwnedMatrix2D::from_vec(c, [2, 2]);

        gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);

        assert_eq!(c.index_item([0, 0]), 7.0);
        assert_eq!(c.index_item([0, 1]), 10.0);
        assert_eq!(c.index_item([1, 0]), 15.0);
        assert_eq!(c.index_item([1, 1]), 22.0);
    }

    #[test]
    fn transposed() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = vec![0.0, 0.0, 0.0, 0.0];

        let mut a = OwnedMatrix2D::from_vec(a, [2, 2]);
        let mut b = OwnedMatrix2D::from_vec(b, [2, 2]);
        let mut c = OwnedMatrix2D::from_vec(c, [2, 2]);

        a.transpose();
        b.transpose();

        gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);

        assert_eq!(c.index_item([0, 0]), 7.0);
        assert_eq!(c.index_item([0, 1]), 15.0);
        assert_eq!(c.index_item([1, 0]), 10.0);
        assert_eq!(c.index_item([1, 1]), 22.0);
    }

    #[test]
    fn gemm_3x4d_4x2d() {
        let x = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let output = vec![0.; 6];

        let x = OwnedMatrix2D::from_vec(x, [3, 4]);
        let y = OwnedMatrix2D::from_vec(y, [4, 2]);
        let mut output = OwnedMatrix2D::from_vec(output, [3, 2]);

        gemm(x.to_view(), y.to_view(), output.to_view_mut(), 1.0, 0.0);

        println!("{:?}", output);

        assert_eq!(output.index_item([0, 0]), 50.0);
        assert_eq!(output.index_item([0, 1]), 60.0);
        assert_eq!(output.index_item([1, 0]), 114.0);
        assert_eq!(output.index_item([1, 1]), 140.0);
        assert_eq!(output.index_item([2, 0]), 178.0);
        assert_eq!(output.index_item([2, 1]), 220.0);
    }

    #[test]
    #[should_panic(
        expected = "The number of columns of matrix A must match the number of rows of matrix B."
    )]
    fn mismatched_shapes() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = vec![0.0, 0.0, 0.0, 0.0];

        let a = OwnedMatrix2D::from_vec(a, [2, 2]);
        let b = OwnedMatrix2D::from_vec(b, [3, 2]);
        let mut c = OwnedMatrix2D::from_vec(c, [2, 2]);

        gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "The output matrix C must not be transposed.")]
    fn transposed_output() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = vec![0.0, 0.0, 0.0, 0.0];

        let a = OwnedMatrix2D::from_vec(a, [2, 2]);
        let b = OwnedMatrix2D::from_vec(b, [2, 2]);
        let mut c = OwnedMatrix2D::from_vec(c, [2, 2]);

        c.transpose();

        gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);
    }
}

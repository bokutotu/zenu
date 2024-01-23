use crate::{
    blas::{Blas, BlasLayout, BlasTrans},
    dim_impl::Dim2,
    matrix::{MatrixBase, ViewMatrix, ViewMutMatix},
    num::Num,
};

pub fn gemm<
    T: Num,
    A: ViewMatrix + MatrixBase<Dim = Dim2, Item = T>,
    B: ViewMatrix + MatrixBase<Dim = Dim2, Item = T>,
    C: ViewMutMatix + MatrixBase<Dim = Dim2, Item = T>,
>(
    a: A,
    b: B,
    c: C,
    alpha: T,
    beta: T,
) {
    let mut c = c;
    let c_shape = c.shape();
    let a_shape = a.shape();
    let b_shape = b.shape();

    // check if transposed
    let is_transposed_a = a.shape_stride().is_transposed();
    let is_transposed_b = b.shape_stride().is_transposed();
    let is_transposed_c = c.shape_stride().is_transposed();

    assert!(!is_transposed_c);

    // shake shape
    // check m
    assert_eq!(a_shape[0], c_shape[0]);
    // check k
    assert_eq!(b_shape[0], a_shape[1]);
    // check n
    assert_eq!(b_shape[1], c_shape[1]);

    let m = a_shape[0];
    let k = b_shape[0];
    let n = b_shape[1];

    // check a stride is default stride and lead dimension's stride is m
    let lead_dim_stride_a = if is_transposed_a {
        a.stride()[1]
    } else {
        a.stride()[0]
    };
    assert!(a.is_default_stride() || a.is_transposed_default_stride());
    assert_eq!(lead_dim_stride_a, m);

    // check b stride is default stride and lead dimension's stride is k
    let lead_dim_stride_b = if is_transposed_b {
        b.stride()[1]
    } else {
        b.stride()[0]
    };
    assert!(b.is_default_stride() || a.is_transposed_default_stride());
    assert_eq!(lead_dim_stride_b, k);

    // check c stride is default stride and lead dimension's stride is m
    let lead_dim_stride_c = if is_transposed_c {
        c.stride()[1]
    } else {
        c.stride()[0]
    };
    assert!(c.is_default_stride());
    assert_eq!(lead_dim_stride_c, m);

    let transa = if is_transposed_a {
        BlasTrans::Ordinary
    } else {
        BlasTrans::None
    };

    let transb = if is_transposed_b {
        BlasTrans::Ordinary
    } else {
        BlasTrans::None
    };

    A::Blas::gemm(
        BlasLayout::RowMajor,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a.as_ptr(),
        lead_dim_stride_a,
        b.as_ptr(),
        lead_dim_stride_b,
        beta,
        c.as_mut_ptr(),
        lead_dim_stride_c,
    );
}

#[cfg(test)]
mod gemm {
    use crate::{
        dim,
        matrix::{IndexItem, MatrixBase, MatrixSlice, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::CpuOwnedMatrix2D,
        operation::transpose::Transpose,
        slice,
    };

    use super::gemm;

    #[test]
    fn non_transposed() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = vec![0.0, 0.0, 0.0, 0.0];

        let a = CpuOwnedMatrix2D::from_vec(a, dim!(2, 2));
        let b = CpuOwnedMatrix2D::from_vec(b, dim!(2, 2));
        let mut c = CpuOwnedMatrix2D::from_vec(c, dim!(2, 2));

        gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);

        assert_eq!(c.index_item(dim!(0, 0)), 7.0);
        assert_eq!(c.index_item(dim!(0, 1)), 10.0);
        assert_eq!(c.index_item(dim!(1, 0)), 15.0);
        assert_eq!(c.index_item(dim!(1, 1)), 22.0);
    }

    #[test]
    fn transposed() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = vec![0.0, 0.0, 0.0, 0.0];

        let mut a = CpuOwnedMatrix2D::from_vec(a, dim!(2, 2));
        let mut b = CpuOwnedMatrix2D::from_vec(b, dim!(2, 2));
        let mut c = CpuOwnedMatrix2D::from_vec(c, dim!(2, 2));

        a.transpose();
        b.transpose();

        a.shape_stride();
        b.shape_stride();

        gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);

        assert_eq!(c.index_item(dim!(0, 0)), 7.0);
        assert_eq!(c.index_item(dim!(0, 1)), 15.0);
        assert_eq!(c.index_item(dim!(1, 0)), 10.0);
        assert_eq!(c.index_item(dim!(1, 1)), 22.0);
    }

    #[test]
    #[should_panic]
    fn sliced() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let c = vec![0.0, 0.0, 0.0, 0.0];

        let a = CpuOwnedMatrix2D::from_vec(a, dim!(2, 4));
        let b = CpuOwnedMatrix2D::from_vec(b, dim!(2, 4));
        let mut c = CpuOwnedMatrix2D::from_vec(c, dim!(2, 2));

        let a = a.slice(slice!(.., ..2));
        let b = b.slice(slice!(.., ..2));

        gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);
    }
}

use crate::{
    blas::{Blas, BlasLayout, BlasTrans},
    dim::Dim2,
    matrix::{MatrixBase, ViewMatrix, ViewMutMatix},
    num::Num,
};

pub fn gemm<T, A, B, C>(a: A, b: B, c: C, alpha: T, beta: T)
where
    T: Num,
    A: ViewMatrix + MatrixBase<Dim = Dim2, Item = T>,
    B: ViewMatrix + MatrixBase<Dim = Dim2, Item = T>,
    C: ViewMutMatix + MatrixBase<Dim = Dim2, Item = T>,
{
    let mut c = c;
    let c_shape = c.shape();
    let a_shape = a.shape();
    let b_shape = b.shape();

    // check if transposed
    let is_transposed_a = a.shape_stride().is_transposed();
    let is_transposed_b = b.shape_stride().is_transposed();
    let is_transposed_c = c.shape_stride().is_transposed();

    assert!(!is_transposed_c);

    let m = a_shape[1];
    let n = b_shape[0];
    let k = a_shape[0];

    // shape check
    assert_eq!(a_shape[1], c_shape[1]);
    assert_eq!(b_shape[0], c_shape[0]);
    assert_eq!(a_shape[0], b_shape[1]);

    let get_inner_stride = |stride: Dim2, is_transpose| {
        if is_transpose {
            stride[0]
        } else {
            stride[1]
        }
    };
    let get_leading_dim = |shape: Dim2, is_transpose| {
        if is_transpose {
            shape[1]
        } else {
            shape[0]
        }
    };

    let leading_dim_a = get_leading_dim(a_shape, is_transposed_a);
    let leading_dim_b = get_leading_dim(b_shape, is_transposed_b);
    let leading_dim_c = get_leading_dim(c_shape, is_transposed_c);

    let inner_stride_a = get_inner_stride(a.stride(), is_transposed_a);
    assert!(a.is_default_stride() || a.is_transposed_default_stride());
    assert_eq!(inner_stride_a, 1);

    let inner_stride_b = get_inner_stride(b.stride(), is_transposed_b);
    assert!(b.is_default_stride() || a.is_transposed_default_stride());
    assert_eq!(inner_stride_b, 1);

    let inner_stride_c = get_inner_stride(c.stride(), is_transposed_c);
    assert!(c.is_default_stride());
    assert_eq!(inner_stride_c, 1);

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
        leading_dim_a,
        b.as_ptr(),
        leading_dim_b,
        beta,
        c.as_mut_ptr(),
        leading_dim_c,
    );
}

#[cfg(test)]
mod gemm {
    use crate::{
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

        let a = CpuOwnedMatrix2D::from_vec(a, &[2, 2]);
        let b = CpuOwnedMatrix2D::from_vec(b, &[2, 2]);
        let mut c = CpuOwnedMatrix2D::from_vec(c, &[2, 2]);

        gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);

        assert_eq!(c.index_item(&[0, 0]), 7.0);
        assert_eq!(c.index_item(&[0, 1]), 10.0);
        assert_eq!(c.index_item([1, 0]), 15.0);
        assert_eq!(c.index_item([1, 1]), 22.0);
    }

    #[test]
    fn transposed() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = vec![0.0, 0.0, 0.0, 0.0];

        let mut a = CpuOwnedMatrix2D::from_vec(a, [2, 2]);
        let mut b = CpuOwnedMatrix2D::from_vec(b, [2, 2]);
        let mut c = CpuOwnedMatrix2D::from_vec(c, [2, 2]);

        a.transpose();
        b.transpose();

        a.shape_stride();
        b.shape_stride();

        gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);

        assert_eq!(c.index_item([0, 0]), 7.0);
        assert_eq!(c.index_item([0, 1]), 15.0);
        assert_eq!(c.index_item([1, 0]), 10.0);
        assert_eq!(c.index_item([1, 1]), 22.0);
    }

    #[test]
    #[should_panic]
    fn sliced() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let c = vec![0.0, 0.0, 0.0, 0.0];

        let a = CpuOwnedMatrix2D::from_vec(a, [2, 4]);
        let b = CpuOwnedMatrix2D::from_vec(b, [2, 4]);
        let mut c = CpuOwnedMatrix2D::from_vec(c, [2, 2]);

        let a = a.slice(slice!(.., ..2));
        let b = b.slice(slice!(.., ..2));

        gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);
    }
}

use crate::{
    blas::{Blas, BlasLayout, BlasTrans},
    dim::{Dim2, DimTrait},
    index::Index0D,
    matrix::{
        IndexAxisDyn, IndexAxisMutDyn, MatrixBase, ToViewMatrix, ToViewMutMatrix, ViewMatrix,
        ViewMutMatix,
    },
    matrix_impl::{matrix_into_dim, Matrix},
    memory::{ToViewMemory, View, ViewMut},
    num::Num,
};

fn get_leading_dim(shape: Dim2, is_transpose: bool) -> usize {
    if is_transpose {
        shape[0]
    } else {
        shape[1]
    }
}

pub(crate) fn gemm_shape_check<A, B, C>(a: &A, b: &B, c: &C) -> Result<(), String>
where
    A: MatrixBase,
    B: MatrixBase,
    C: MatrixBase,
{
    let c_shape = c.shape();
    let a_shape = a.shape();
    let b_shape = b.shape();

    if c_shape.len() != 2 {
        return Err("The output matrix C must be 2-D.".to_string());
    }
    if a_shape.len() != 2 {
        return Err("The input matrix A must be 2-D.".to_string());
    }
    if b_shape.len() != 2 {
        return Err("The input matrix B must be 2-D.".to_string());
    }

    let is_transposed_c = c.shape_stride().is_transposed();

    if is_transposed_c {
        return Err("The output matrix C must not be transposed.".to_string());
    }

    if a_shape[0] != c_shape[0] {
        return Err(
            "The number of rows of matrix A must match the number of rows of matrix C.".to_string(),
        );
    }

    if b_shape[1] != c_shape[1] {
        return Err(
            "The number of columns of matrix B must match the number of columns of matrix C."
                .to_string(),
        );
    }

    if a_shape[1] != b_shape[0] {
        return Err(
            "The number of columns of matrix A must match the number of rows of matrix B."
                .to_string(),
        );
    }

    if a_shape[0] == 0
        || a_shape[1] == 0
        || b_shape[0] == 0
        || b_shape[1] == 0
        || c_shape[0] == 0
        || c_shape[1] == 0
    {
        return Err(
            "The dimensions of the input and output matrices must be greater than 0.".to_string(),
        );
    }
    Ok(())
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
    gemm_shape_check(&a, &b, &c).unwrap();
    gemm_unchecked(a, b, c, alpha, beta);
}

pub(crate) fn gemm_batch_shape_check<AM, BM, CM, AD, BD, CD>(
    a: &Matrix<AM, AD>,
    b: &Matrix<BM, BD>,
    c: &Matrix<CM, CD>,
) -> Result<(), String>
where
    AM: View + ToViewMemory,
    BM: View + ToViewMemory,
    CM: ViewMut + ToViewMemory,
    AD: DimTrait,
    BD: DimTrait,
    CD: DimTrait,
{
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape();

    // Check dimensions
    let min_dim = 2;
    let max_dim = 3;
    if a_shape.len() < min_dim || b_shape.len() < min_dim || c_shape.len() < min_dim {
        return Err("The input and output matrices must be at least 2-D.".to_string());
    }
    if a_shape.len() > max_dim || b_shape.len() > max_dim || c_shape.len() > max_dim {
        return Err("The input and output matrices must be at most 3-D.".to_string());
    }

    // Check broadcast dimensions
    if a_shape.len() == max_dim && b_shape.len() == max_dim && c_shape.len() == max_dim {
        if a_shape[0] != b_shape[0] || a_shape[0] != c_shape[0] {
            return Err(format!(
                "Mismatched batch dimensions: a.shape() = {:?}, b.shape() = {:?}, c.shape() = {:?}",
                a_shape, b_shape, c_shape
            ));
        }
    }

    let a_dyn = a.to_view().into_dyn_dim();
    let a_view = if a_dyn.shape().len() == 3 {
        a_dyn.index_axis_dyn(Index0D::new(0))
    } else {
        a_dyn.to_view()
    };

    let b_dyn = b.to_view().into_dyn_dim();
    let b_view = if b_dyn.shape().len() == 3 {
        b_dyn.index_axis_dyn(Index0D::new(0))
    } else {
        b_dyn.to_view()
    };

    let c_dyn = c.to_view().into_dyn_dim();
    let c_view = if c_dyn.shape().len() == 3 {
        c_dyn.index_axis_dyn(Index0D::new(0))
    } else {
        c_dyn.to_view()
    };

    gemm_shape_check(&a_view, &b_view, &c_view)?;
    Ok(())
}

pub(crate) fn gemm_batch_unchecked<T, AM, BM, CM, AD, BD, CD>(
    a: Matrix<AM, AD>,
    b: Matrix<BM, BD>,
    c: Matrix<CM, CD>,
    alpha: T,
    beta: T,
) where
    T: Num,
    AM: View + ToViewMemory<Item = T>,
    BM: View + ToViewMemory<Item = T>,
    CM: ViewMut + ToViewMemory<Item = T>,
    AD: DimTrait,
    BD: DimTrait,
    CD: DimTrait,
{
    let a = a.into_dyn_dim();
    let b = b.into_dyn_dim();
    let mut c = c.into_dyn_dim();

    for idx in 0..c.shape()[0] {
        let a = if a.shape().len() == 3 {
            a.index_axis_dyn(Index0D::new(idx))
        } else {
            a.to_view()
        };
        let b = if b.shape().len() == 3 {
            b.index_axis_dyn(Index0D::new(idx))
        } else {
            b.to_view()
        };
        let c = if c.shape().len() == 3 {
            c.index_axis_mut_dyn(Index0D::new(idx))
        } else {
            c.to_view_mut()
        };

        let a = matrix_into_dim(a);
        let b = matrix_into_dim(b);
        let c = matrix_into_dim(c);

        gemm_unchecked(a, b, c, alpha, beta);
    }
}

pub fn gemm_batch<T, AM, BM, CM, AD, BD, CD>(
    a: Matrix<AM, AD>,
    b: Matrix<BM, BD>,
    c: Matrix<CM, CD>,
    alpha: T,
    beta: T,
) where
    T: Num,
    AM: View + ToViewMemory<Item = T>,
    BM: View + ToViewMemory<Item = T>,
    CM: ViewMut + ToViewMemory<Item = T>,
    AD: DimTrait,
    BD: DimTrait,
    CD: DimTrait,
{
    gemm_batch_shape_check(&a, &b, &c).unwrap();
    gemm_batch_unchecked(a, b, c, alpha, beta);
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

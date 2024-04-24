use crate::{
    device::{cpu::Cpu, DeviceBase},
    dim::{Dim2, DimTrait},
    index::Index0D,
    matrix::{Matrix, Ref, Repr},
    num::Num,
    shape_stride::ShapeStride,
};

use super::{BlasLayout, BlasTrans};

pub trait Gemm: DeviceBase {
    #[allow(clippy::too_many_arguments)]
    fn gemm_raw<T: Num>(
        layout: BlasLayout,
        transa: BlasTrans,
        transb: BlasTrans,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *const T,
        ldb: usize,
        beta: T,
        c: *mut T,
        ldc: usize,
    );

    fn gemm_shape_check<SA, SB, SC>(
        a: ShapeStride<SA>,
        b: ShapeStride<SB>,
        c: ShapeStride<SC>,
    ) -> Result<(), String>
    where
        SA: DimTrait,
        SB: DimTrait,
        SC: DimTrait;

    fn gemm_unchecked<T, RA, RB>(
        a: Matrix<RA, Dim2, Self>,
        b: Matrix<RB, Dim2, Self>,
        c: Matrix<Ref<&mut T>, Dim2, Self>,
        alpha: T,
        beta: T,
    ) where
        T: Num,
        RA: Repr<Item = T>,
        RB: Repr<Item = T>;

    fn gemm<T, RA, RB>(
        a: Matrix<RA, Dim2, Self>,
        b: Matrix<RB, Dim2, Self>,
        c: Matrix<Ref<&mut T>, Dim2, Self>,
        alpha: T,
        beta: T,
    ) where
        T: Num,
        RA: Repr<Item = T>,
        RB: Repr<Item = T>,
    {
        Self::gemm_shape_check(a.shape_stride(), b.shape_stride(), c.shape_stride()).unwrap();
        Self::gemm_unchecked(a, b, c, alpha, beta);
    }
}

impl Gemm for Cpu {
    fn gemm_raw<T: Num>(
        layout: BlasLayout,
        transa: BlasTrans,
        transb: BlasTrans,
        m: usize,
        n: usize,
        k: usize,
        alpha: T,
        a: *const T,
        lda: usize,
        b: *const T,
        ldb: usize,
        beta: T,
        c: *mut T,
        ldc: usize,
    ) {
        extern crate openblas_src;
        use cblas::*;
        fn from_trans(value: BlasTrans) -> Transpose {
            match value {
                BlasTrans::None => Transpose::None,
                BlasTrans::Ordinary => Transpose::Ordinary,
                BlasTrans::Conjugate => Transpose::Conjugate,
            }
        }

        fn from_layout(value: BlasLayout) -> Layout {
            match value {
                BlasLayout::RowMajor => Layout::RowMajor,
                BlasLayout::ColMajor => Layout::ColumnMajor,
            }
        }

        let layout = from_layout(layout);
        let transa = from_trans(transa);
        let transb = from_trans(transb);

        if T::is_f32() {
            let a = unsafe { std::slice::from_raw_parts(a as *const f32, lda) };
            let b = unsafe { std::slice::from_raw_parts(b as *const f32, ldb) };
            let c = unsafe { std::slice::from_raw_parts_mut(c as *mut f32, ldc) };

            let m = m.try_into().unwrap();
            let n = n.try_into().unwrap();
            let k = k.try_into().unwrap();
            let lda = lda.try_into().unwrap();
            let ldb = ldb.try_into().unwrap();
            let ldc = ldc.try_into().unwrap();

            unsafe {
                sgemm(
                    layout,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    *(&alpha as *const T as *const f32),
                    a,
                    lda,
                    b,
                    ldb,
                    *(&beta as *const T as *const f32),
                    c,
                    ldc,
                )
            }
        } else {
            let a = unsafe { std::slice::from_raw_parts(a as *const f64, lda) };
            let b = unsafe { std::slice::from_raw_parts(b as *const f64, ldb) };
            let c = unsafe { std::slice::from_raw_parts_mut(c as *mut f64, ldc) };

            let m = m.try_into().unwrap();
            let n = n.try_into().unwrap();
            let k = k.try_into().unwrap();
            let lda = lda.try_into().unwrap();
            let ldb = ldb.try_into().unwrap();
            let ldc = ldc.try_into().unwrap();

            unsafe {
                dgemm(
                    layout,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    *(&alpha as *const T as *const f64),
                    a,
                    lda,
                    b,
                    ldb,
                    *(&beta as *const T as *const f64),
                    c,
                    ldc,
                )
            }
        }
    }

    fn gemm_shape_check<SA, SB, SC>(
        a: ShapeStride<SA>,
        b: ShapeStride<SB>,
        c: ShapeStride<SC>,
    ) -> Result<(), String>
    where
        SA: DimTrait,
        SB: DimTrait,
        SC: DimTrait,
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

        let is_transposed_c = c.is_transposed();

        if is_transposed_c {
            return Err("The output matrix C must not be transposed.".to_string());
        }

        if a_shape[0] != c_shape[0] {
            return Err(
                "The number of rows of matrix A must match the number of rows of matrix C."
                    .to_string(),
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
                "The dimensions of the input and output matrices must be greater than 0."
                    .to_string(),
            );
        }
        Ok(())
    }

    fn gemm_unchecked<T, RA, RB>(
        a: Matrix<RA, Dim2, Self>,
        b: Matrix<RB, Dim2, Self>,
        c: Matrix<Ref<&mut T>, Dim2, Self>,
        alpha: T,
        beta: T,
    ) where
        T: Num,
        RA: Repr<Item = T>,
        RB: Repr<Item = T>,
    {
        fn get_leading_dim(shape: Dim2, is_transpose: bool) -> usize {
            if is_transpose {
                shape[0]
            } else {
                shape[1]
            }
        }

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

        Self::gemm_raw(
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
}

pub fn gemm<T, D, RA, RB>(
    a: Matrix<RA, Dim2, D>,
    b: Matrix<RB, Dim2, D>,
    c: Matrix<Ref<&mut T>, Dim2, D>,
    alpha: T,
    beta: T,
) where
    T: Num,
    D: Gemm,
    RA: Repr<Item = T>,
    RB: Repr<Item = T>,
{
    D::gemm(a, b, c, alpha, beta);
}

// #[cfg(test)]
// mod gemm {
//     use crate::{
//         matrix::{IndexItem, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
//         operation::transpose::Transpose,
//     };
//
//     use super::gemm;
//
//     #[test]
//     fn non_transposed() {
//         let a = vec![1.0, 2.0, 3.0, 4.0];
//         let b = vec![1.0, 2.0, 3.0, 4.0];
//         let c = vec![0.0, 0.0, 0.0, 0.0];
//
//         let a = OwnedMatrix2D::from_vec(a, [2, 2]);
//         let b = OwnedMatrix2D::from_vec(b, [2, 2]);
//         let mut c = OwnedMatrix2D::from_vec(c, [2, 2]);
//
//         gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);
//
//         assert_eq!(c.index_item([0, 0]), 7.0);
//         assert_eq!(c.index_item([0, 1]), 10.0);
//         assert_eq!(c.index_item([1, 0]), 15.0);
//         assert_eq!(c.index_item([1, 1]), 22.0);
//     }
//
//     #[test]
//     fn transposed() {
//         let a = vec![1.0, 2.0, 3.0, 4.0];
//         let b = vec![1.0, 2.0, 3.0, 4.0];
//         let c = vec![0.0, 0.0, 0.0, 0.0];
//
//         let mut a = OwnedMatrix2D::from_vec(a, [2, 2]);
//         let mut b = OwnedMatrix2D::from_vec(b, [2, 2]);
//         let mut c = OwnedMatrix2D::from_vec(c, [2, 2]);
//
//         a.transpose();
//         b.transpose();
//
//         gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);
//
//         assert_eq!(c.index_item([0, 0]), 7.0);
//         assert_eq!(c.index_item([0, 1]), 15.0);
//         assert_eq!(c.index_item([1, 0]), 10.0);
//         assert_eq!(c.index_item([1, 1]), 22.0);
//     }
//
//     #[test]
//     fn gemm_3x4d_4x2d() {
//         let x = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
//         let y = vec![1., 2., 3., 4., 5., 6., 7., 8.];
//         let output = vec![0.; 6];
//
//         let x = OwnedMatrix2D::from_vec(x, [3, 4]);
//         let y = OwnedMatrix2D::from_vec(y, [4, 2]);
//         let mut output = OwnedMatrix2D::from_vec(output, [3, 2]);
//
//         gemm(x.to_view(), y.to_view(), output.to_view_mut(), 1.0, 0.0);
//
//         assert_eq!(output.index_item([0, 0]), 50.0);
//         assert_eq!(output.index_item([0, 1]), 60.0);
//         assert_eq!(output.index_item([1, 0]), 114.0);
//         assert_eq!(output.index_item([1, 1]), 140.0);
//         assert_eq!(output.index_item([2, 0]), 178.0);
//         assert_eq!(output.index_item([2, 1]), 220.0);
//     }
//
//     #[test]
//     #[should_panic(
//         expected = "The number of columns of matrix A must match the number of rows of matrix B."
//     )]
//     fn mismatched_shapes() {
//         let a = vec![1.0, 2.0, 3.0, 4.0];
//         let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//         let c = vec![0.0, 0.0, 0.0, 0.0];
//
//         let a = OwnedMatrix2D::from_vec(a, [2, 2]);
//         let b = OwnedMatrix2D::from_vec(b, [3, 2]);
//         let mut c = OwnedMatrix2D::from_vec(c, [2, 2]);
//
//         gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);
//     }
//
//     #[test]
//     #[should_panic(expected = "The output matrix C must not be transposed.")]
//     fn transposed_output() {
//         let a = vec![1.0, 2.0, 3.0, 4.0];
//         let b = vec![1.0, 2.0, 3.0, 4.0];
//         let c = vec![0.0, 0.0, 0.0, 0.0];
//
//         let a = OwnedMatrix2D::from_vec(a, [2, 2]);
//         let b = OwnedMatrix2D::from_vec(b, [2, 2]);
//         let mut c = OwnedMatrix2D::from_vec(c, [2, 2]);
//
//         c.transpose();
//
//         gemm(a.to_view(), b.to_view(), c.to_view_mut(), 1.0, 1.0);
//     }
// }

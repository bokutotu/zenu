use cblas::Transpose;

use crate::{
    device::{cpu::Cpu, Device, DeviceBase},
    dim::DimTrait,
    matrix::{Matrix, Ref, Repr},
    matrix_blas::BlasTrans,
    num::Num,
    shape_stride::ShapeStride,
};

pub trait Gemm: DeviceBase {
    fn gemm_unchecked<T: Num>(
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

    // fn gemm_shape_check<SA: DimTrait, SB: DimTrait, SC: DimTrait>(
    //     a: ShapeStride<SA>,
    //     b: ShapeStride<SB>,
    //     c: ShapeStride<SC>,
    // ) -> Result<(), String>;
}

fn from_trans(value: BlasTrans) -> Transpose {
    match value {
        BlasTrans::None => Transpose::None,
        BlasTrans::Ordinary => Transpose::Ordinary,
        BlasTrans::Conjugate => Transpose::Conjugate,
    }
}

impl Gemm for Cpu {
    fn gemm_unchecked<T: Num>(
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
        if T::is_f32() {
            let a = unsafe { std::slice::from_raw_parts(a as *const f32, m * k) };
            let b = unsafe { std::slice::from_raw_parts(b as *const f32, k * n) };
            let c = unsafe { std::slice::from_raw_parts_mut(c as *mut f32, m * n) };
            unsafe {
                sgemm(
                    Layout::RowMajor,
                    from_trans(transa),
                    from_trans(transb),
                    m.try_into().unwrap(),
                    n.try_into().unwrap(),
                    k.try_into().unwrap(),
                    alpha.to_f32().unwrap(),
                    a,
                    lda.try_into().unwrap(),
                    b,
                    ldb.try_into().unwrap(),
                    beta.to_f32().unwrap(),
                    c,
                    ldc.try_into().unwrap(),
                )
            }
        } else {
            let a = unsafe { std::slice::from_raw_parts(a as *const f64, m * k) };
            let b = unsafe { std::slice::from_raw_parts(b as *const f64, k * n) };
            let c = unsafe { std::slice::from_raw_parts_mut(c as *mut f64, m * n) };
            unsafe {
                dgemm(
                    Layout::RowMajor,
                    from_trans(transa),
                    from_trans(transb),
                    m.try_into().unwrap(),
                    n.try_into().unwrap(),
                    k.try_into().unwrap(),
                    alpha.to_f64().unwrap(),
                    a,
                    lda.try_into().unwrap(),
                    b,
                    ldb.try_into().unwrap(),
                    beta.to_f64().unwrap(),
                    c,
                    ldc.try_into().unwrap(),
                )
            }
        }
    }
}

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

#[cfg(feature = "nvidia")]
use zenu_cuda::cublas::{cublas_gemm, ZenuCublasOperation};

#[cfg(feature = "nvidia")]
impl Gemm for Nvidia {
    fn gemm_unchecked<T: Num>(
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
        fn to_cuda_ops(trans: BlasTrans) -> ZenuCublasOperation {
            match trans {
                BlasTrans::None => ZenuCublasOperation::N,
                BlasTrans::Ordinary => ZenuCublasOperation::T,
                BlasTrans::Conjugate => ZenuCublasOperation::ConjT,
            }
        }
        let transa = to_cuda_ops(transa);
        let transb = to_cuda_ops(transb);
        let m = m as i32;
        let n = n as i32;
        let k = k as i32;
        let lda = lda as i32;
        let ldb = ldb as i32;
        let ldc = ldc as i32;
        unsafe {
            cublas_gemm::<T>(transb, transa, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc).unwrap();
        }
    }
}

fn gemm_shape_check<SA: DimTrait, SB: DimTrait, SC: DimTrait>(
    a: ShapeStride<SA>,
    b: ShapeStride<SB>,
    c: ShapeStride<SC>,
) -> Result<(), String> {
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

pub fn gemm<T, D, RA, RB, SA, SB, SC>(
    a: &Matrix<RA, SA, D>,
    b: &Matrix<RB, SB, D>,
    c: &Matrix<Ref<&mut T>, SC, D>,
    alpha: T,
    beta: T,
) where
    T: Num,
    D: Device,
    RA: Repr<Item = T>,
    RB: Repr<Item = T>,
    SA: DimTrait,
    SB: DimTrait,
    SC: DimTrait,
{
    if let Ok(()) = gemm_shape_check(a.shape_stride(), b.shape_stride(), c.shape_stride()) {
        D::gemm_unchecked(
            BlasTrans::None,
            BlasTrans::None,
            c.shape()[0],
            c.shape()[1],
            a.shape()[1],
            alpha,
            a.as_ptr(),
            a.stride()[0],
            b.as_ptr(),
            b.stride()[0],
            beta,
            c.as_mut_ptr(),
            c.stride()[0],
        );
        return;
    }
    panic!("Dimension mismatch");
}

#[cfg(test)]
mod gemm {
    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    use super::gemm;

    fn gemm_3x4_4x5_3x5<D: Device>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [3, 4],
        );
        let b = Matrix::<_, DimDyn, D>::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20.,
            ],
            [4, 5],
        );
        let mut c = Matrix::<_, DimDyn, D>::zeros([3, 5]);
        gemm(&a, &b, &c.to_ref_mut(), 1., 0.);
        let ans = vec![
            110., 120., 130., 140., 150., 246., 272., 298., 324., 350., 382., 424., 466., 508.,
            550.,
        ];
        let ans = Matrix::<_, DimDyn, D>::from_vec(ans, [3, 5]);
        let diff = (c - ans).asum();
        assert!(diff < 1e-6);
    }
    #[test]
    fn gemm_3x4_4x5_3x5_cpu() {
        gemm_3x4_4x5_3x5::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn gemm_3x4_4x5_3x5_nvidia() {
        gemm_3x4_4x5_3x5::<crate::device::nvidia::Nvidia>();
    }
}

// use crate::{
//     dim::DimTrait,
//     matrix::{ToViewMatrix, ToViewMutMatrix},
//     matrix_blas::gemm::{
//         gemm_batch_shape_check, gemm_batch_unchecked, gemm_shape_check, gemm_unchecked,
//     },
//     matrix_impl::{matrix_into_dim, Matrix},
//     memory::{ToViewMemory, ViewMut},
//     num::Num,
// };
//
// /// Trait for computing the General Matrix Multiply (GEMM) operation.
// ///
// /// The `gemm` function performs a matrix multiplication operation.
// /// It takes two matrices as input and multiplies them together, storing the result in `self`.
// ///
// /// # Shape Requirements
// ///
// /// - `self`: The output matrix, must be a 2-D matrix.
// /// - `rhs`: The right-hand side input matrix, must be a 2-D matrix.
// /// - `lhs`: The left-hand side input matrix, must be a 2-D matrix.
// ///
// /// The shapes of the input matrices must satisfy the following conditions:
// /// - The number of columns of `rhs` must match the number of rows of `lhs`.
// /// - The number of rows of `self` must match the number of rows of `rhs`.
// /// - The number of columns of `self` must match the number of columns of `lhs`.
// ///
// /// If the input matrices are higher-dimensional (3-D or more), the leading dimensions are
// /// treated as batch dimensions, and the last two dimensions are used for matrix multiplication.
// ///
// /// # Panics
// ///
// /// This function will panic if:
// /// - The shapes of the input matrices do not satisfy the above conditions.
// /// - The dimensions of the input and output matrices are not greater than zero.
// ///
// /// # Examples
// ///
// /// ```
// /// use zenu_matrix::{
// ///     matrix::{IndexItem, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
// ///     matrix_impl::OwnedMatrix2D,
// ///     constructor::zeros::Zeros,
// /// };
// ///
// /// use zenu_matrix::operation::mul::Gemm;
// ///
// /// let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
// /// let b = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], [3, 4]);
// /// let mut ans = OwnedMatrix2D::<f32>::zeros([2, 4]);
// ///
// /// ans.to_view_mut().gemm(a.to_view(), b.to_view());
// ///
// /// assert_eq!(ans.index_item([0, 0]), 38.);
// /// assert_eq!(ans.index_item([0, 1]), 44.);
// /// assert_eq!(ans.index_item([0, 2]), 50.);
// /// assert_eq!(ans.index_item([0, 3]), 56.);
// /// assert_eq!(ans.index_item([1, 0]), 83.);
// /// assert_eq!(ans.index_item([1, 1]), 98.);
// /// assert_eq!(ans.index_item([1, 2]), 113.);
// /// assert_eq!(ans.index_item([1, 3]), 128.);
// /// ```
// pub trait Gemm<Rhs, Lhs>: ToViewMutMatrix {
//     /// Performs the General Matrix Multiply (GEMM) operation.
//     ///
//     /// This function takes two matrices as input and multiplies them together, storing the result in `self`.
//     ///
//     /// # Arguments
//     ///
//     /// * `rhs` - The right-hand side matrix.
//     /// * `lhs` - The left-hand side matrix.
//     ///
//     /// # Panics
//     ///
//     /// This function will panic if the dimensions of the matrices do not allow for matrix multiplication.
//     fn gemm(self, rhs: Rhs, lhs: Lhs);
// }
//
// impl<T, M1, M2, M3, D1, D2, D3> Gemm<Matrix<M1, D1>, Matrix<M2, D2>> for Matrix<M3, D3>
// where
//     T: Num,
//     D1: DimTrait,
//     D2: DimTrait,
//     D3: DimTrait,
//     M1: ToViewMemory<Item = T>,
//     M2: ToViewMemory<Item = T>,
//     M3: ViewMut<Item = T>,
// {
//     fn gemm(self, rhs: Matrix<M1, D1>, lhs: Matrix<M2, D2>) {
//         // したのコードをif let Ok(())に続く形で書き直して
//         let rhs = rhs.to_view();
//         let lhs = lhs.to_view();
//         if let Ok(()) = gemm_shape_check(&rhs, &lhs, &self) {
//             gemm_unchecked(
//                 matrix_into_dim(rhs),
//                 matrix_into_dim(lhs),
//                 matrix_into_dim(self),
//                 T::one(),
//                 T::zero(),
//             );
//             return;
//         }
//         if let Ok(()) = gemm_batch_shape_check(&rhs, &lhs, &self) {
//             gemm_batch_unchecked(rhs, lhs, self, T::one(), T::zero());
//             return;
//         }
//
//         panic!("Dimension mismatch");
//     }
// }
//
// #[cfg(test)]
// mod mat_mul {
//     use crate::{
//         constructor::zeros::Zeros,
//         matrix::{IndexItem, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
//         matrix_impl::{OwnedMatrix2D, OwnedMatrix3D},
//         operation::transpose::Transpose,
//     };
//
//     use super::*;
//
//     #[test]
//     fn default() {
//         let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
//         let b = OwnedMatrix2D::from_vec(
//             vec![
//                 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
//             ],
//             [3, 5],
//         );
//         let mut ans = OwnedMatrix2D::<f32>::zeros([2, 5]);
//
//         ans.to_view_mut().gemm(a.to_view(), b.to_view());
//         assert_eq!(ans.index_item([0, 0]), 46.);
//         assert_eq!(ans.index_item([0, 1]), 52.);
//         assert_eq!(ans.index_item([0, 2]), 58.);
//         assert_eq!(ans.index_item([0, 3]), 64.);
//         assert_eq!(ans.index_item([0, 4]), 70.);
//         assert_eq!(ans.index_item([1, 0]), 100.);
//         assert_eq!(ans.index_item([1, 1]), 115.);
//         assert_eq!(ans.index_item([1, 2]), 130.);
//         assert_eq!(ans.index_item([1, 3]), 145.);
//         assert_eq!(ans.index_item([1, 4]), 160.);
//     }
//
//     #[test]
//     fn default_stride_2() {
//         let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
//         // shape 3 4
//         let b = OwnedMatrix2D::from_vec(
//             vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
//             [3, 4],
//         );
//         let mut ans = OwnedMatrix2D::<f32>::zeros([2, 4]);
//
//         ans.to_view_mut().gemm(a.to_view(), b.to_view());
//
//         assert_eq!(ans.index_item([0, 0]), 38.);
//         assert_eq!(ans.index_item([0, 1]), 44.);
//         assert_eq!(ans.index_item([0, 2]), 50.);
//         assert_eq!(ans.index_item([0, 3]), 56.);
//         assert_eq!(ans.index_item([1, 0]), 83.);
//         assert_eq!(ans.index_item([1, 1]), 98.);
//         assert_eq!(ans.index_item([1, 2]), 113.);
//         assert_eq!(ans.index_item([1, 3]), 128.);
//     }
//
//     #[test]
//     fn gemm_2d() {
//         let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], [2, 2]);
//         let b = OwnedMatrix2D::from_vec(vec![5., 6., 7., 8.], [2, 2]);
//         let mut c = OwnedMatrix2D::<f32>::zeros([2, 2]);
//
//         c.to_view_mut().gemm(a.to_view(), b.to_view());
//
//         assert_eq!(c.index_item([0, 0]), 19.);
//         assert_eq!(c.index_item([0, 1]), 22.);
//         assert_eq!(c.index_item([1, 0]), 43.);
//         assert_eq!(c.index_item([1, 1]), 50.);
//     }
//
//     #[test]
//     fn gemm_3d() {
//         let a = OwnedMatrix3D::from_vec(
//             vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
//             [2, 2, 3],
//         );
//         let b = OwnedMatrix3D::from_vec(
//             vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
//             [2, 3, 2],
//         );
//         let mut c = OwnedMatrix3D::<f32>::zeros([2, 2, 2]);
//
//         c.to_view_mut().gemm(a.to_view(), b.to_view());
//
//         assert_eq!(c.index_item([0, 0, 0]), 22.);
//         assert_eq!(c.index_item([0, 0, 1]), 28.);
//         assert_eq!(c.index_item([0, 1, 0]), 49.);
//         assert_eq!(c.index_item([0, 1, 1]), 64.);
//         assert_eq!(c.index_item([1, 0, 0]), 220.);
//         assert_eq!(c.index_item([1, 0, 1]), 244.);
//         assert_eq!(c.index_item([1, 1, 0]), 301.);
//         assert_eq!(c.index_item([1, 1, 1]), 334.);
//     }
//
//     #[test]
//     fn gemm_transposed_a() {
//         let mut a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], [2, 2]);
//         let b = OwnedMatrix2D::from_vec(vec![5., 6., 7., 8.], [2, 2]);
//         let mut c = OwnedMatrix2D::<f32>::zeros([2, 2]);
//
//         a.transpose();
//         c.to_view_mut().gemm(a.to_view(), b.to_view());
//
//         assert_eq!(c.index_item([0, 0]), 26.);
//         assert_eq!(c.index_item([0, 1]), 30.);
//         assert_eq!(c.index_item([1, 0]), 38.);
//         assert_eq!(c.index_item([1, 1]), 44.);
//     }
//
//     #[test]
//     fn gemm_transposed_b() {
//         let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], [2, 2]);
//         let mut b = OwnedMatrix2D::from_vec(vec![5., 6., 7., 8.], [2, 2]);
//         let mut c = OwnedMatrix2D::<f32>::zeros([2, 2]);
//
//         b.transpose();
//         c.to_view_mut().gemm(a.to_view(), b.to_view());
//
//         assert_eq!(c.index_item([0, 0]), 17.);
//         assert_eq!(c.index_item([0, 1]), 23.);
//         assert_eq!(c.index_item([1, 0]), 39.);
//         assert_eq!(c.index_item([1, 1]), 53.);
//     }
//
//     #[test]
//     fn gemm_transposed_a_and_b() {
//         let mut a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], [2, 2]);
//         let mut b = OwnedMatrix2D::from_vec(vec![5., 6., 7., 8.], [2, 2]);
//         let mut c = OwnedMatrix2D::<f32>::zeros([2, 2]);
//
//         a.transpose();
//         b.transpose();
//         c.to_view_mut().gemm(a.to_view(), b.to_view());
//
//         assert_eq!(c.index_item([0, 0]), 23.);
//         assert_eq!(c.index_item([0, 1]), 31.);
//         assert_eq!(c.index_item([1, 0]), 34.);
//         assert_eq!(c.index_item([1, 1]), 46.);
//     }
//
//     #[test]
//     #[should_panic(expected = "Dimension mismatch")]
//     fn gemm_dimension_mismatch() {
//         let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], [2, 2]);
//         let b = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]);
//         let mut c = OwnedMatrix2D::<f32>::zeros([2, 2]);
//
//         c.to_view_mut().gemm(a.to_view(), b.to_view());
//     }
// }

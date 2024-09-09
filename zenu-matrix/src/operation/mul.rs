use cblas::Transpose;

use crate::{
    device::{cpu::Cpu, Device, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref, Repr},
    matrix_blas::BlasTrans,
    num::Num,
    shape_stride::ShapeStride,
};

pub trait Gemm: DeviceBase {
    #[expect(clippy::too_many_arguments)]
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
}

fn from_trans(value: BlasTrans) -> Transpose {
    match value {
        BlasTrans::None => Transpose::None,
        BlasTrans::Ordinary => Transpose::Ordinary,
        BlasTrans::Conjugate => Transpose::Conjugate,
    }
}

impl Gemm for Cpu {
    #[expect(clippy::many_single_char_names, clippy::similar_names)]
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
        use cblas::{dgemm, sgemm, Layout};
        if T::is_f32() {
            let a = unsafe { std::slice::from_raw_parts(a.cast(), m * k) };
            let b = unsafe { std::slice::from_raw_parts(b.cast(), k * n) };
            let c = unsafe { std::slice::from_raw_parts_mut(c.cast(), m * n) };
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
                );
            }
        } else {
            let a = unsafe { std::slice::from_raw_parts(a.cast(), m * k) };
            let b = unsafe { std::slice::from_raw_parts(b.cast(), k * n) };
            let c = unsafe { std::slice::from_raw_parts_mut(c.cast(), m * n) };
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
                );
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
    #[expect(clippy::many_single_char_names, clippy::similar_names)]
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
        let m = i32::try_from(m).unwrap();
        let n = i32::try_from(n).unwrap();
        let k = i32::try_from(k).unwrap();
        let lda = i32::try_from(lda).unwrap();
        let ldb = i32::try_from(ldb).unwrap();
        let ldc = i32::try_from(ldc).unwrap();
        cublas_gemm::<T>(transb, transa, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc).unwrap();
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

#[expect(clippy::missing_panics_doc, clippy::similar_names)]
pub fn gemm_assign<T, D, RA, RB, SA, SB, SC>(
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
        let transa = if a.shape_stride().is_transposed() {
            BlasTrans::Ordinary
        } else {
            BlasTrans::None
        };
        let transb = if b.shape_stride().is_transposed() {
            BlasTrans::Ordinary
        } else {
            BlasTrans::None
        };
        let get_lead_dim = |stride: &[usize], trans: BlasTrans| match trans {
            BlasTrans::None => stride[0],
            BlasTrans::Ordinary => stride[1],
            BlasTrans::Conjugate => unreachable!(),
        };
        let lda = get_lead_dim(a.stride().slice(), transa);
        let ldb = get_lead_dim(b.stride().slice(), transb);
        D::gemm_unchecked(
            transa,
            transb,
            c.shape()[0],
            c.shape()[1],
            a.shape()[1],
            alpha,
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            c.stride()[0],
        );
        return;
    }
    panic!("Dimension mismatch");
}

pub fn gemm<T, D, RA, RB, SA, SB>(
    a: &Matrix<RA, SA, D>,
    b: &Matrix<RB, SB, D>,
    alpha: T,
    beta: T,
) -> Matrix<Owned<T>, DimDyn, D>
where
    T: Num,
    D: Device,
    RA: Repr<Item = T>,
    RB: Repr<Item = T>,
    SA: DimTrait,
    SB: DimTrait,
{
    let c_shape = [a.shape()[0], b.shape()[1]];
    let mut c = Matrix::<_, DimDyn, D>::alloc(c_shape);
    gemm_assign(a, b, &c.to_ref_mut(), alpha, beta);
    c
}

pub fn matmul<T, D, RA, RB, SA, SB>(
    a: &Matrix<RA, SA, D>,
    b: &Matrix<RB, SB, D>,
) -> Matrix<Owned<T>, DimDyn, D>
where
    T: Num,
    D: Device,
    RA: Repr<Item = T>,
    RB: Repr<Item = T>,
    SA: DimTrait,
    SB: DimTrait,
{
    gemm(a, b, T::one(), T::zero())
}

#[cfg(test)]
mod gemm {
    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    use super::gemm_assign;

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
        let mut c = Matrix::<_, DimDyn, D>::alloc([3, 5]);
        gemm_assign(&a, &b, &c.to_ref_mut(), 1., 0.);
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

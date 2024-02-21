use crate::{
    blas::Blas,
    dim::{Dim2, DimTrait},
    index::Index0D,
    matrix::{
        AsMutPtr, IndexAxisDyn, IndexAxisMutDyn, IndexItem, IndexItemAsign, MatrixBase, ViewMatrix,
        ViewMutMatix,
    },
    matrix_blas::gemm::gemm,
    matrix_impl::Matrix,
    memory::{ViewMemory, ViewMutMemory},
    num::Num,
    operation::copy_from::CopyFrom,
};

fn mul_matrix_scalar<T, LM, SM, D>(self_: Matrix<SM, D>, lhs: Matrix<LM, D>, rhs: T)
where
    T: Num,
    SM: ViewMutMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    D: DimTrait,
{
    assert_eq!(self_.shape(), lhs.shape());
    let mut self_ = self_.into_dyn_dim();
    let lhs = lhs.into_dyn_dim();
    CopyFrom::copy_from(&mut self_, &lhs);
    mul_assign_matrix_scalar(self_, rhs);
}

fn mul_assign_matrix_scalar<T, LM, D>(to: Matrix<LM, D>, other: T)
where
    T: Num,
    LM: ViewMutMemory<Item = T>,
    D: DimTrait,
{
    let mut to = to.into_dyn_dim();

    macro_rules! scal {
        () => {{
            if to.shape_stride().is_contiguous() {
                let num_dim = to.shape().len();
                LM::Blas::scal(
                    to.shape().num_elm(),
                    other,
                    to.as_mut_ptr(),
                    to.stride()[num_dim - 1],
                );
            } else {
                for idx in 0..to.shape()[0] {
                    let to_ = to.index_axis_mut_dyn(Index0D::new(idx));
                    mul_assign_matrix_scalar(to_, other);
                }
            }
        }};
    }

    match to.shape().len() {
        0 => to.index_item_asign([], to.index_item([]) * other),
        1 => LM::Blas::scal(to.shape().num_elm(), other, to.as_mut_ptr(), to.stride()[0]),
        2 => scal!(),
        3 => scal!(),
        4 => scal!(),
        _ => panic!("not implemented: this is bug. please report this bug."),
    };
}

fn mul_matrix_matrix<T, LM, RM, SM, D1, D2, D3>(
    self_: Matrix<SM, D1>,
    lhs: Matrix<LM, D2>,
    rhs: Matrix<RM, D3>,
) where
    T: Num,
    LM: ViewMemory<Item = T>,
    RM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
    D3: DimTrait,
{
    if lhs.shape().len() < rhs.shape().len() {
        mul_matrix_matrix(self_, rhs, lhs);
        return;
    }
    assert_eq!(self_.shape().slice(), lhs.shape().slice());
    let mut self_ = self_.into_dyn_dim();
    let lhs = lhs.into_dyn_dim();
    self_.copy_from(&lhs);
    mul_assign_matrix_matrix(self_, rhs);
}
fn mul_assign_matrix_matrix<T, TM, OM, D1, D2>(to: Matrix<TM, D1>, other: Matrix<OM, D2>)
where
    T: Num,
    TM: ViewMutMemory<Item = T>,
    OM: ViewMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
{
    let mut to = to.into_dyn_dim();
    let other = other.into_dyn_dim();

    assert!(to.shape().is_include(&other.shape()));
    if to.shape().is_empty() {
        to.index_item_asign([], other.index_item([]) * other.index_item([]));
        return;
    }
    if other.shape().is_empty() {
        let scalar = other.index_item([]);
        mul_assign_matrix_scalar(to, scalar);
        return;
    }

    if to.shape().len() == 1 {
        for i in 0..to.shape()[0] {
            let t_itm = to.index_item([i]);
            let o_itm = other.index_item([i]);
            to.index_item_asign([i], t_itm * o_itm);
        }
    } else if to.shape() == other.shape() {
        for i in 0..to.shape()[0] {
            let to_ = to.index_axis_mut_dyn(Index0D::new(i));
            let other = other.index_axis_dyn(Index0D::new(i));
            mul_assign_matrix_matrix(to_, other);
        }
    } else {
        for i in 0..to.shape()[0] {
            let to_ = to.index_axis_mut_dyn(Index0D::new(i));
            mul_assign_matrix_matrix(to_, other.clone());
        }
    }
}

pub trait MatrixMul<Lhs, Rhs>: ViewMutMatix {
    fn mul(self, lhs: Lhs, rhs: Rhs);
}

impl<T, RM, SM, D> MatrixMul<Matrix<RM, D>, T> for Matrix<SM, D>
where
    T: Num,
    RM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
    D: DimTrait,
{
    fn mul(self, lhs: Matrix<RM, D>, rhs: T) {
        mul_matrix_scalar(self, lhs, rhs);
    }
}

impl<T, DS, DR, SM, LM, RM> MatrixMul<Matrix<LM, DS>, Matrix<RM, DR>> for Matrix<SM, DS>
where
    T: Num,
    DS: DimTrait,
    DR: DimTrait,
    SM: ViewMutMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    RM: ViewMemory<Item = T>,
{
    fn mul(self, lhs: Matrix<LM, DS>, rhs: Matrix<RM, DR>) {
        mul_matrix_matrix(self, lhs, rhs);
    }
}

pub trait MatMul<Rhs, Lhs>: ViewMutMatix {
    fn mat_mul(self, rhs: Rhs, lhs: Lhs);
}

impl<T, S, R, L> MatMul<R, L> for S
where
    T: Num,
    L: ViewMatrix<Item = T> + MatrixBase<Dim = Dim2>,
    R: ViewMatrix<Item = T> + MatrixBase<Dim = Dim2>,
    S: ViewMutMatix<Item = T> + MatrixBase<Dim = Dim2>,
{
    fn mat_mul(self, rhs: R, lhs: L) {
        gemm(rhs, lhs, self, T::one(), T::zero());
    }
}

#[cfg(test)]
mod mul {
    use crate::{
        matrix::{IndexItem, MatrixSlice, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::{CpuOwnedMatrix0D, CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix4D},
        operation::zeros::Zeros,
        slice,
    };

    use super::MatrixMul;

    #[test]
    fn mul_1d_scalar() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b = CpuOwnedMatrix0D::from_vec(vec![2.0], []);
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros([3]);
        ans.to_view_mut().mul(a.to_view(), b.to_view());

        assert_eq!(ans.index_item([0]), 2.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 6.0);
    }

    #[test]
    fn scalar_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3.], [3]);
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros([3]);
        ans.to_view_mut().mul(a.to_view(), 2.);

        assert_eq!(ans.index_item([0]), 2.);
        assert_eq!(ans.index_item([1]), 4.);
        assert_eq!(ans.index_item([2]), 6.);
    }

    #[test]
    fn sliced_scalar_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], [4]);
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros([2]);
        ans.to_view_mut().mul(a.to_view().slice(slice!(..;2)), 2.);

        assert_eq!(ans.index_item([0]), 2.);
        assert_eq!(ans.index_item([1]), 6.);
    }

    #[test]
    fn scalar_2d() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let mut ans = CpuOwnedMatrix2D::<f32>::zeros([2, 3]);
        ans.to_view_mut().mul(a.to_view(), 2.);

        assert_eq!(ans.index_item([0, 0]), 2.);
        assert_eq!(ans.index_item([0, 1]), 4.);
        assert_eq!(ans.index_item([0, 2]), 6.);
        assert_eq!(ans.index_item([1, 0]), 8.);
        assert_eq!(ans.index_item([1, 1]), 10.);
        assert_eq!(ans.index_item([1, 2]), 12.);
    }

    #[test]
    fn default_1d_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3.], [3]);
        let b = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3.], [3]);
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros([3]);
        ans.to_view_mut().mul(a.to_view(), b.to_view());

        assert_eq!(ans.index_item([0]), 1.);
        assert_eq!(ans.index_item([1]), 4.);
        assert_eq!(ans.index_item([2]), 9.);
    }

    #[test]
    fn sliced_1d_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], [4]);
        let b = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], [4]);
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros([2]);
        ans.to_view_mut().mul(
            a.to_view().slice(slice!(..;2)),
            b.to_view().slice(slice!(..;2)),
        );

        assert_eq!(ans.index_item([0]), 1.);
        assert_eq!(ans.index_item([1]), 9.);
    }

    #[test]
    fn default_2d_2d() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let b = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let mut ans = CpuOwnedMatrix2D::<f32>::zeros([2, 3]);
        ans.to_view_mut().mul(a.to_view(), b.to_view());

        assert_eq!(ans.index_item([0, 0]), 1.);
        assert_eq!(ans.index_item([0, 1]), 4.);
        assert_eq!(ans.index_item([0, 2]), 9.);
        assert_eq!(ans.index_item([1, 0]), 16.);
        assert_eq!(ans.index_item([1, 1]), 25.);
        assert_eq!(ans.index_item([1, 2]), 36.);
    }

    #[test]
    fn sliced_4d_2d() {
        let mut a_vec = Vec::new();
        for i in 0..2 * 2 * 2 * 2 {
            a_vec.push(i as f32);
        }

        let a = CpuOwnedMatrix4D::from_vec(a_vec, [2, 2, 2, 2]);
        let b = CpuOwnedMatrix1D::from_vec(vec![1., 2.], [2]);

        let mut ans = CpuOwnedMatrix4D::<f32>::zeros([2, 2, 2, 2]);

        ans.to_view_mut().mul(a.to_view(), b.to_view());

        println!("{:?}", ans);

        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        assert_eq!(
                            ans.index_item([i, j, k, l]),
                            a.index_item([i, j, k, l]) * b.index_item([l])
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod mat_mul {
    use crate::{
        matrix::{IndexItem, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::CpuOwnedMatrix2D,
        operation::zeros::Zeros,
    };

    use super::*;

    #[test]
    fn default() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]);
        let b = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let mut ans = CpuOwnedMatrix2D::<f32>::zeros([2, 2]);

        ans.to_view_mut().mat_mul(a.to_view(), b.to_view());
        dbg!(ans.index_item([0, 0]));
        dbg!(ans.index_item([0, 1]));
        dbg!(ans.index_item([1, 0]));
        dbg!(ans.index_item([1, 1]));
        assert_eq!(ans.index_item([0, 0]), 22.);
        assert_eq!(ans.index_item([0, 1]), 28.);
        assert_eq!(ans.index_item([1, 0]), 49.);
        assert_eq!(ans.index_item([1, 1]), 64.);
    }

    #[test]
    fn default_stride_2() {
        let a = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]);
        // shape 3 4
        let b = CpuOwnedMatrix2D::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [4, 3],
        );
        let mut ans = CpuOwnedMatrix2D::<f32>::zeros([4, 2]);

        ans.to_view_mut().mat_mul(a.to_view(), b.to_view());
        dbg!(ans.index_item([0, 0]));
        dbg!(ans.index_item([0, 1]));
        dbg!(ans.index_item([1, 0]));
        dbg!(ans.index_item([1, 1]));
        dbg!(ans.index_item([2, 0]));
        dbg!(ans.index_item([2, 1]));
        dbg!(ans.index_item([3, 0]));
        dbg!(ans.index_item([3, 1]));

        assert_eq!(ans.index_item([0, 0]), 38.);
        assert_eq!(ans.index_item([0, 1]), 44.);
        assert_eq!(ans.index_item([1, 0]), 50.);
        assert_eq!(ans.index_item([1, 1]), 56.);
        assert_eq!(ans.index_item([2, 0]), 83.);
        assert_eq!(ans.index_item([2, 1]), 98.);
        assert_eq!(ans.index_item([3, 0]), 113.);
        assert_eq!(ans.index_item([3, 1]), 128.);
    }
}

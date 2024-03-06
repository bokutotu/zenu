use crate::{
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{
        IndexAxisDyn, IndexAxisMutDyn, MatrixBase, OwnedMatrix, ToOwnedMatrix, ToViewMatrix,
        ToViewMutMatrix,
    },
    matrix_impl::Matrix,
    memory::{ToOwnedMemory, ToViewMemory, ToViewMutMemory},
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
};

pub trait MatrixDiv<L> {
    type Output: OwnedMatrix;
    fn div(self, lhs: L) -> Self::Output;
}

pub trait MatrixDivAssign<L, R> {
    fn div_assign(&mut self, lhs: L, rhs: R);
}

fn div_1d_1d_cpu<T: Num, D1: DimTrait, D2: DimTrait, D3: DimTrait>(
    to: &mut Matrix<ViewMutMem<T>, D1>,
    a: &Matrix<ViewMem<T>, D2>,
    b: &Matrix<ViewMem<T>, D3>,
) {
    let num_elm = to.shape().num_elm();
    let to_stride = to.stride()[0];
    let a_stride = a.stride()[0];
    let b_stride = b.stride()[0];
    let slice_to = to.as_mut_slice();
    let slice_a = a.as_slice();
    let slice_b = b.as_slice();
    for i in 0..num_elm {
        slice_to[i * to_stride] = slice_a[i * a_stride] / slice_b[i * b_stride];
    }
}

fn div_1d_scalar_cpu<T: Num, D1: DimTrait, D2: DimTrait>(
    to: &mut Matrix<ViewMutMem<T>, D1>,
    a: &Matrix<ViewMem<T>, D2>,
    b: T,
) {
    let num_elm = to.shape().num_elm();
    let to_stride = to.stride()[0];
    let a_stride = a.stride()[0];
    let slice_to = to.as_mut_slice();
    let slice_a = a.as_slice();
    for i in 0..num_elm {
        slice_to[i * to_stride] = slice_a[i * a_stride] / b;
    }
}

impl<T, D1, D2, M1, M2> MatrixDivAssign<Matrix<M1, D1>, T> for Matrix<M2, D2>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMutMemory<Item = T>,
{
    fn div_assign(&mut self, lhs: Matrix<M1, D1>, rhs: T) {
        if self.shape().slice() != lhs.shape().slice() {
            panic!("Matrix shape mismatch");
        }

        if self.shape().is_empty() {
            self.as_mut_slice()[0] = lhs.as_slice()[0] / rhs;
        } else if self.shape().len() == 1 {
            div_1d_scalar_cpu(&mut self.to_view_mut(), &lhs.to_view(), rhs);
        } else {
            let num_iter = self.shape()[0];
            for idx in 0..num_iter {
                let mut s = self.index_axis_mut_dyn(Index0D::new(idx));
                let lhs = lhs.index_axis_dyn(Index0D::new(idx));
                s.div_assign(lhs, rhs);
            }
        }
    }
}

impl<T: Num, D: DimTrait, M: ToViewMemory<Item = T> + ToOwnedMemory> MatrixDiv<T> for Matrix<M, D> {
    type Output = Matrix<M::Owned, D>;
    fn div(self, lhs: T) -> Self::Output {
        let mut owned = self.to_owned();
        let s = self.into_dyn_dim();
        let s_v = s.to_view();
        owned.to_view_mut().div_assign(s_v, lhs);
        owned
    }
}

impl<T, D1, D2, D3, M1, M2, M3> MatrixDivAssign<Matrix<M1, D1>, Matrix<M2, D2>> for Matrix<M3, D3>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    D3: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMemory<Item = T>,
    M3: ToViewMutMemory<Item = T>,
{
    fn div_assign(&mut self, lhs: Matrix<M1, D1>, rhs: Matrix<M2, D2>) {
        if lhs.shape().len() < rhs.shape().len() {
            panic!("Matrix shape mismatch");
        }

        if self.shape().slice() != lhs.shape().slice() {
            panic!("Matrix shape mismatch");
        }

        let self_shape_dyn = DimDyn::from(self.shape().slice());
        let rhs_shape_dyn = DimDyn::from(rhs.shape().slice());
        if !self_shape_dyn.is_include(&rhs_shape_dyn) {
            panic!("Matrix shape mismatch");
        }

        if rhs.shape().is_empty() {
            self.div_assign(lhs, rhs.as_slice()[0]);
            return;
        }

        if self.shape().is_empty() {
            self.as_mut_slice()[0] = lhs.as_slice()[0] / rhs.as_slice()[0];
        } else if self.shape().len() == 1 {
            div_1d_1d_cpu(&mut self.to_view_mut(), &lhs.to_view(), &rhs.to_view());
        } else {
            let num_iter = self.shape()[0];
            for idx in 0..num_iter {
                let mut s = self.index_axis_mut_dyn(Index0D::new(idx));
                let lhs = lhs.index_axis_dyn(Index0D::new(idx));
                let rhs = rhs.index_axis_dyn(Index0D::new(idx));
                s.div_assign(lhs, rhs);
            }
        }
    }
}

// impl<T: Num, D: DimTrait, M1: ToViewMemory + ToOwnedMemory + ToViewMutMemory<Item = T>>
//     MatrixDiv<Matrix<M1, D>> for Matrix<M1, D>
impl<T, M1, M2, D1, D2> MatrixDiv<Matrix<M1, D1>> for Matrix<M2, D2>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMemory<Item = T> + ToOwnedMemory,
{
    type Output = Matrix<M2::Owned, D2>;
    fn div(self, lhs: Matrix<M1, D1>) -> Self::Output {
        let mut owned = self.to_owned();
        let s = self.into_dyn_dim();
        let s_v = s.to_view();
        owned.to_view_mut().div_assign(s_v, lhs.to_view());
        owned
    }
}

#[cfg(test)]
mod div {
    use crate::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::{asum::Asum, div::MatrixDiv},
    };

    #[test]
    fn div_0d_0d() {
        let a = OwnedMatrixDyn::from_vec(vec![2.0], &[]);
        let b = OwnedMatrixDyn::from_vec(vec![3.0], &[]);
        let c = a.div(b);
        assert_eq!(c.as_slice(), &[2.0 / 3.0]);
    }

    #[test]
    fn div_1d_0d() {
        let a = OwnedMatrixDyn::from_vec(vec![2.0, 3.0], &[2]);
        let b = OwnedMatrixDyn::from_vec(vec![3.0], &[]);
        let c = a.div(b);
        assert_eq!(c.as_slice(), &[2.0 / 3.0, 3.0 / 3.0]);
    }

    #[test]
    fn div_3d_3d() {
        let a = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let b = OwnedMatrixDyn::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[2, 2, 2]);
        let c = a.div(b);
        let ans = vec![
            1.0 / 2.0,
            2.0 / 3.0,
            3.0 / 4.0,
            4.0 / 5.0,
            5.0 / 6.0,
            6.0 / 7.0,
            7.0 / 8.0,
            8.0 / 9.0,
        ];
        let ans = OwnedMatrixDyn::from_vec(ans, &[2, 2, 2]);
        let diff = c.to_view() - ans.to_view();
        let asum = diff.asum();
        assert_eq!(asum, 0.0);
    }
}
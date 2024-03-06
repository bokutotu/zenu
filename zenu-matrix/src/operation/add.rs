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

pub trait MatrixAdd<L> {
    type Output: OwnedMatrix;
    fn add(self, lhs: L) -> Self::Output;
}

pub trait MatrixAddAssign<L, R> {
    fn add_assign(&mut self, lhs: L, rhs: R);
}

fn add_1d_1d_cpu<T: Num, D1: DimTrait, D2: DimTrait, D3: DimTrait>(
    to: &mut Matrix<ViewMutMem<T>, D1>,
    lhs: Matrix<ViewMem<T>, D2>,
    rhs: Matrix<ViewMem<T>, D3>,
) {
    let num_elm = to.shape().num_elm();
    let inner_slice_to = to.stride()[to.shape().len() - 1];
    let inner_slice_lhs = lhs.stride()[lhs.shape().len() - 1];
    let inner_slice_rhs = rhs.stride()[rhs.shape().len() - 1];
    let to_slice = to.as_mut_slice();
    let lhs_slice = lhs.as_slice();
    let rhs_slice = rhs.as_slice();
    for i in 0..num_elm {
        to_slice[i * inner_slice_to] =
            lhs_slice[i * inner_slice_lhs] + rhs_slice[i * inner_slice_rhs];
    }
}

fn add_1d_scalar_cpu<T: Num, D1: DimTrait, D2: DimTrait>(
    to: &mut Matrix<ViewMutMem<T>, D1>,
    lhs: Matrix<ViewMem<T>, D2>,
    rhs: T,
) {
    let num_elm = to.shape().num_elm();
    let inner_slice_to = to.stride()[to.shape().len() - 1];
    let inner_slice_lhs = lhs.stride()[lhs.shape().len() - 1];
    let to_slice = to.as_mut_slice();
    let lhs_slice = lhs.as_slice();
    for i in 0..num_elm {
        to_slice[i * inner_slice_to] = lhs_slice[i * inner_slice_lhs] + rhs;
    }
}

impl<T, D1, D2, M1, M2> MatrixAddAssign<Matrix<M1, D1>, T> for Matrix<M2, D2>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMutMemory<Item = T>,
{
    fn add_assign(&mut self, lhs: Matrix<M1, D1>, rhs: T) {
        if self.shape().slice() != lhs.shape().slice() {
            panic!("Matrix shape mismatch");
        }

        if self.shape().is_empty() {
            self.as_mut_slice()[0] = lhs.as_slice()[0] / rhs;
        } else if self.shape().len() == 1 {
            add_1d_scalar_cpu(&mut self.to_view_mut(), lhs.to_view(), rhs);
        } else {
            let num_iter = self.shape()[0];
            for idx in 0..num_iter {
                let mut s = self.index_axis_mut_dyn(Index0D::new(idx));
                let lhs = lhs.index_axis_dyn(Index0D::new(idx));
                s.add_assign(lhs, rhs);
            }
        }
    }
}

impl<T, D, M> MatrixAdd<T> for Matrix<M, D>
where
    T: Num,
    D: DimTrait,
    M: ToViewMemory<Item = T> + ToOwnedMemory,
{
    type Output = Matrix<M::Owned, D>;
    fn add(self, lhs: T) -> Self::Output {
        let mut owned = self.to_owned();
        let s = self.into_dyn_dim();
        let s_v = s.to_view();
        owned.to_view_mut().add_assign(s_v, lhs);
        owned
    }
}

impl<T, D1, D2, D3, M1, M2, M3> MatrixAddAssign<Matrix<M1, D1>, Matrix<M2, D2>> for Matrix<M3, D3>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    D3: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMemory<Item = T>,
    M3: ToViewMutMemory<Item = T>,
{
    fn add_assign(&mut self, lhs: Matrix<M1, D1>, rhs: Matrix<M2, D2>) {
        if lhs.shape().len() < rhs.shape().len() {
            self.add_assign(rhs, lhs);
            return;
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
            self.add_assign(lhs, rhs.as_slice()[0]);
            return;
        }

        if self.shape().is_empty() {
            self.as_mut_slice()[0] = lhs.as_slice()[0] / rhs.as_slice()[0];
        } else if self.shape().len() == 1 {
            add_1d_1d_cpu(&mut self.to_view_mut(), lhs.to_view(), rhs.to_view());
        } else {
            let self_dim_len = self.shape().len();
            let rhs_dim_len = rhs.shape().len();
            let num_iter = self.shape()[0];
            for idx in 0..num_iter {
                let mut s = self.index_axis_mut_dyn(Index0D::new(idx));
                let lhs = lhs.index_axis_dyn(Index0D::new(idx));
                let rhs = if self_dim_len == rhs_dim_len {
                    rhs.index_axis_dyn(Index0D::new(idx))
                } else if self_dim_len > rhs_dim_len {
                    rhs.to_view().into_dyn_dim()
                } else {
                    panic!("this is bug");
                };
                s.add_assign(lhs, rhs);
            }
        }
    }
}

// impl<T: Num, D: DimTrait, M1: ToViewMemory + ToOwnedMemory + ToViewMutMemory<Item = T>>
//     MatrixDiv<Matrix<M1, D>> for Matrix<M1, D>
impl<T, M1, M2, D1, D2> MatrixAdd<Matrix<M1, D1>> for Matrix<M2, D2>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMemory<Item = T> + ToOwnedMemory,
{
    type Output = Matrix<M2::Owned, D2>;
    fn add(self, lhs: Matrix<M1, D1>) -> Self::Output {
        let mut owned = self.to_owned();
        let s = self.into_dyn_dim();
        let s_v = s.to_view();
        owned.to_view_mut().add_assign(s_v, lhs.to_view());
        owned
    }
}

#[cfg(test)]
mod add {
    use crate::{
        matrix::{IndexItem, MatrixSlice, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::{OwnedMatrix0D, OwnedMatrix1D, OwnedMatrix2D, OwnedMatrix3D, OwnedMatrixDyn},
        operation::zeros::Zeros,
        slice,
    };

    use super::*;

    #[test]
    fn add_dyn_dyn() {
        let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let ans = OwnedMatrix1D::<f32>::zeros([3]);

        let a = a.into_dyn_dim();
        let b = b.into_dyn_dim();
        let mut ans = ans.into_dyn_dim();

        ans.to_view_mut().add_assign(a.to_view(), b.to_view());
    }

    #[test]
    fn add_1d_scalar() {
        let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let mut ans = OwnedMatrix1D::<f32>::zeros([3]);
        let b = OwnedMatrix0D::from_vec(vec![2.0], []);
        ans.to_view_mut().add_assign(a.to_view(), b.to_view());

        assert_eq!(ans.index_item([0]), 3.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 5.0);
    }

    #[test]
    fn add_1d_scalar_default_stride() {
        let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let mut ans = OwnedMatrix1D::<f32>::zeros([3]);
        ans.to_view_mut().add_assign(a.to_view(), 1.0);

        assert_eq!(ans.index_item([0]), 2.0);
        assert_eq!(ans.index_item([1]), 3.0);
        assert_eq!(ans.index_item([2]), 4.0);
    }

    #[test]
    fn add_1d_scalar_sliced() {
        let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
        let mut ans = OwnedMatrix1D::<f32>::zeros([3]);

        let sliced = a.slice(slice!(..;2));

        ans.to_view_mut().add_assign(sliced.to_view(), 1.0);

        assert_eq!(ans.index_item([0]), 2.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 6.0);
    }

    #[test]
    fn add_3d_scalar_sliced() {
        let a = OwnedMatrix3D::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
            ],
            [3, 3, 4],
        );

        let mut ans = OwnedMatrix3D::<f32>::zeros([3, 3, 2]);

        let sliced = a.slice(slice!(.., .., ..;2));

        ans.to_view_mut().add_assign(sliced.to_view(), 1.0);

        assert_eq!(ans.index_item([0, 0, 0]), 2.0);
        assert_eq!(ans.index_item([0, 0, 1]), 4.0);
        assert_eq!(ans.index_item([0, 1, 0]), 6.0);
        assert_eq!(ans.index_item([0, 1, 1]), 8.0);
        assert_eq!(ans.index_item([0, 2, 0]), 10.0);
        assert_eq!(ans.index_item([0, 2, 1]), 12.0);
        assert_eq!(ans.index_item([1, 0, 0]), 14.0);
        assert_eq!(ans.index_item([1, 0, 1]), 16.0);
        assert_eq!(ans.index_item([1, 1, 0]), 18.0);
        assert_eq!(ans.index_item([1, 1, 1]), 20.0);
        assert_eq!(ans.index_item([1, 2, 0]), 22.0);
        assert_eq!(ans.index_item([1, 2, 1]), 24.0);
        assert_eq!(ans.index_item([2, 0, 0]), 26.0);
        assert_eq!(ans.index_item([2, 0, 1]), 28.0);
        assert_eq!(ans.index_item([2, 1, 0]), 30.0);
        assert_eq!(ans.index_item([2, 1, 1]), 32.0);
        assert_eq!(ans.index_item([2, 2, 0]), 34.0);
        assert_eq!(ans.index_item([2, 2, 1]), 36.0);
    }

    #[test]
    fn add_1d_1d_default_stride() {
        let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let mut ans = OwnedMatrix1D::<f32>::zeros([3]);
        ans.to_view_mut().add_assign(a.to_view(), b.to_view());

        assert_eq!(ans.index_item([0]), 2.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 6.0);
    }

    #[test]
    fn add_1d_1d_sliced() {
        let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
        let b = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);

        let mut ans = OwnedMatrix1D::<f32>::zeros([3]);

        let sliced_a = a.slice(slice!(..;2));
        let sliced_b = b.slice(slice!(1..;2));

        ans.to_view_mut()
            .add_assign(sliced_a.to_view(), sliced_b.to_view());

        assert_eq!(ans.index_item([0]), 3.0);
        assert_eq!(ans.index_item([1]), 7.0);
        assert_eq!(ans.index_item([2]), 11.0);
    }

    #[test]
    fn add_2d_1d_default() {
        let a = OwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );

        let b = OwnedMatrix1D::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], [8]);

        let mut ans = OwnedMatrix2D::<f32>::zeros([2, 2]);

        let sliced_a = a.slice(slice!(..2, ..2));
        let sliced_b = b.slice(slice!(..2));

        ans.to_view_mut()
            .add_assign(sliced_a.to_view(), sliced_b.to_view());

        assert_eq!(ans.index_item([0, 0]), 2.0);
        assert_eq!(ans.index_item([0, 1]), 4.0);
        assert_eq!(ans.index_item([1, 0]), 6.0);
        assert_eq!(ans.index_item([1, 1]), 8.0);
    }

    #[test]
    fn add_3d_1d_sliced() {
        let mut v = Vec::new();
        let num_elm = 4 * 4 * 4;
        for i in 0..num_elm {
            v.push(i as f32);
        }
        let a = OwnedMatrix3D::from_vec(v, [4, 4, 4]);

        let b = OwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], [4]);

        let mut ans = OwnedMatrix3D::<f32>::zeros([2, 2, 2]);

        let sliced_a = a.slice(slice!(..2, 1..;2, ..2));
        let sliced_b = b.slice(slice!(..2));

        ans.to_view_mut()
            .add_assign(sliced_a.to_view(), sliced_b.to_view());

        assert_eq!(ans.index_item([0, 0, 0]), 5.);
        assert_eq!(ans.index_item([0, 0, 1]), 7.);
        assert_eq!(ans.index_item([0, 1, 0]), 13.);
        assert_eq!(ans.index_item([0, 1, 1]), 15.);
        assert_eq!(ans.index_item([1, 0, 0]), 21.);
        assert_eq!(ans.index_item([1, 0, 1]), 23.);
        assert_eq!(ans.index_item([1, 1, 0]), 29.);
        assert_eq!(ans.index_item([1, 1, 1]), 31.);
    }

    #[test]
    fn add_2d_2d_default() {
        let a = OwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );

        let b = OwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );

        let mut ans = OwnedMatrix2D::<f32>::zeros([4, 4]);
        ans.to_view_mut().add_assign(a.to_view(), b.to_view());

        assert_eq!(ans.index_item([0, 0]), 2.0);
        assert_eq!(ans.index_item([0, 1]), 4.0);
        assert_eq!(ans.index_item([0, 2]), 6.0);
        assert_eq!(ans.index_item([0, 3]), 8.0);
        assert_eq!(ans.index_item([1, 0]), 10.0);
        assert_eq!(ans.index_item([1, 1]), 12.0);
        assert_eq!(ans.index_item([1, 2]), 14.0);
        assert_eq!(ans.index_item([1, 3]), 16.0);
        assert_eq!(ans.index_item([2, 0]), 18.0);
        assert_eq!(ans.index_item([2, 1]), 20.0);
        assert_eq!(ans.index_item([2, 2]), 22.0);
        assert_eq!(ans.index_item([2, 3]), 24.0);
        assert_eq!(ans.index_item([3, 0]), 26.0);
        assert_eq!(ans.index_item([3, 1]), 28.0);
        assert_eq!(ans.index_item([3, 2]), 30.0);
        assert_eq!(ans.index_item([3, 3]), 32.0);
    }

    #[test]
    fn add_2d_0d() {
        let a = OwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );
        let b = OwnedMatrix0D::from_vec(vec![1.], []);
        let mut ans = OwnedMatrix2D::<f32>::zeros([4, 4]);
        ans.to_view_mut().add_assign(a.to_view(), b.to_view());
        assert_eq!(ans.index_item([0, 0]), 2.0);
        assert_eq!(ans.index_item([0, 1]), 3.0);
        assert_eq!(ans.index_item([0, 2]), 4.0);
        assert_eq!(ans.index_item([0, 3]), 5.0);
        assert_eq!(ans.index_item([1, 0]), 6.0);
        assert_eq!(ans.index_item([1, 1]), 7.0);
        assert_eq!(ans.index_item([1, 2]), 8.0);
        assert_eq!(ans.index_item([1, 3]), 9.0);
        assert_eq!(ans.index_item([2, 0]), 10.0);
        assert_eq!(ans.index_item([2, 1]), 11.0);
        assert_eq!(ans.index_item([2, 2]), 12.0);
        assert_eq!(ans.index_item([2, 3]), 13.0);
        assert_eq!(ans.index_item([3, 0]), 14.0);
        assert_eq!(ans.index_item([3, 1]), 15.0);
        assert_eq!(ans.index_item([3, 2]), 16.0);
        assert_eq!(ans.index_item([3, 3]), 17.0);
    }

    #[test]
    fn add_2d_0d_dyn() {
        let a = OwnedMatrixDyn::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );
        let b = OwnedMatrixDyn::from_vec(vec![1.], []);
        let mut ans = OwnedMatrixDyn::<f32>::zeros([4, 4]);
        ans.to_view_mut().add_assign(a.to_view(), b.to_view());
        assert_eq!(ans.index_item([0, 0]), 2.0);
        assert_eq!(ans.index_item([0, 1]), 3.0);
        assert_eq!(ans.index_item([0, 2]), 4.0);
        assert_eq!(ans.index_item([0, 3]), 5.0);
        assert_eq!(ans.index_item([1, 0]), 6.0);
        assert_eq!(ans.index_item([1, 1]), 7.0);
        assert_eq!(ans.index_item([1, 2]), 8.0);
        assert_eq!(ans.index_item([1, 3]), 9.0);
        assert_eq!(ans.index_item([2, 0]), 10.0);
        assert_eq!(ans.index_item([2, 1]), 11.0);
        assert_eq!(ans.index_item([2, 2]), 12.0);
        assert_eq!(ans.index_item([2, 3]), 13.0);
        assert_eq!(ans.index_item([3, 0]), 14.0);
        assert_eq!(ans.index_item([3, 1]), 15.0);
        assert_eq!(ans.index_item([3, 2]), 16.0);
        assert_eq!(ans.index_item([3, 3]), 17.0);
    }

    #[test]
    fn add_4d_2d_dyn() {
        let zeros_4d = OwnedMatrixDyn::<f32>::zeros([2, 2, 2, 2]);
        let ones_2d = OwnedMatrixDyn::from_vec(vec![1., 1., 1., 1.], [2, 2]);
        let mut ans = OwnedMatrixDyn::<f32>::zeros([2, 2, 2, 2]);
        ans.to_view_mut()
            .add_assign(zeros_4d.to_view(), ones_2d.to_view());
    }
}

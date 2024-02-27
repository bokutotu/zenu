use crate::{
    dim::DimTrait,
    matrix::{ToViewMatrix, ViewMutMatix},
    matrix_impl::Matrix,
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
};

use super::{add::MatrixAddAssign, copy_from::CopyFrom};

pub trait MatrixSubAssign<Rhs>: ViewMutMatix {
    fn sub_assign(self, rhs: Rhs);
}

pub trait MatrixSub<Lhs, Rhs>: ViewMutMatix {
    fn sub(self, lhs: Lhs, rhs: Rhs);
}

impl<'a, 'b, T, D1, D2> MatrixSubAssign<Matrix<ViewMem<'a, T>, D1>>
    for Matrix<ViewMutMem<'b, T>, D2>
where
    D1: DimTrait,
    D2: DimTrait,
    T: Num,
{
    fn sub_assign(self, rhs: Matrix<ViewMem<T>, D1>) {
        let rhs = rhs * T::minus_one();
        MatrixAddAssign::add_assign(self, rhs.to_view());
    }
}

impl<'a, T, D> MatrixSubAssign<T> for Matrix<ViewMutMem<'a, T>, D>
where
    D: DimTrait,
    T: Num,
{
    fn sub_assign(self, rhs: T) {
        let rhs = rhs * T::minus_one();
        MatrixAddAssign::add_assign(self, rhs);
    }
}

impl<'a, 'b, 'c, T, D1, D2, D3> MatrixSub<Matrix<ViewMem<'a, T>, D1>, Matrix<ViewMem<'b, T>, D2>>
    for Matrix<ViewMutMem<'b, T>, D3>
where
    D1: DimTrait,
    D2: DimTrait,
    D3: DimTrait,
    T: Num,
{
    fn sub(self, lhs: Matrix<ViewMem<T>, D1>, rhs: Matrix<ViewMem<T>, D2>) {
        let mut self_ = self.into_dyn_dim();
        let lhs = lhs.into_dyn_dim();
        CopyFrom::copy_from(&mut self_, &lhs);
        let rhs = rhs * T::minus_one();
        MatrixSubAssign::sub_assign(self_, rhs.to_view());
    }
}

impl<'a, 'b, T, D1, D2> MatrixSub<Matrix<ViewMem<'a, T>, D1>, T> for Matrix<ViewMutMem<'b, T>, D2>
where
    D1: DimTrait,
    D2: DimTrait,
    T: Num,
{
    fn sub(self, lhs: Matrix<ViewMem<T>, D1>, rhs: T) {
        let mut self_ = self.into_dyn_dim();
        let lhs = lhs.into_dyn_dim();
        CopyFrom::copy_from(&mut self_, &lhs);
        let rhs = rhs * T::minus_one();
        MatrixSubAssign::sub_assign(self_, rhs);
    }
}

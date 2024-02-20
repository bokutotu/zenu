use crate::{
    dim::DimTrait,
    matrix::{ToViewMatrix, ViewMutMatix},
    matrix_impl::Matrix,
    memory::{ViewMemory, ViewMutMemory},
    num::Num,
};

use super::{add::MatrixAddAssign, copy_from::CopyFrom};

pub trait MatrixSubAssign<Rhs>: ViewMutMatix {
    fn sub_assign(self, rhs: Rhs);
}

pub trait MatrixSub<Lhs, Rhs>: ViewMutMatix {
    fn sub(self, lhs: Lhs, rhs: Rhs);
}

impl<T, SM, RM, D1, D2> MatrixSubAssign<Matrix<RM, D1>> for Matrix<SM, D2>
where
    SM: ViewMutMemory<Item = T>,
    RM: ViewMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
    T: Num,
{
    fn sub_assign(self, rhs: Matrix<RM, D1>) {
        let rhs = rhs * T::minus_one();
        MatrixAddAssign::add_assign(self, rhs.to_view());
    }
}

impl<T, SM, D> MatrixSubAssign<T> for Matrix<SM, D>
where
    SM: ViewMutMemory<Item = T>,
    D: DimTrait,
    T: Num,
{
    fn sub_assign(self, rhs: T) {
        let rhs = rhs * T::minus_one();
        MatrixAddAssign::add_assign(self, rhs);
    }
}

impl<T, SM, RM, LM, D1, D2, D3> MatrixSub<Matrix<LM, D1>, Matrix<RM, D2>> for Matrix<SM, D3>
where
    SM: ViewMutMemory<Item = T>,
    RM: ViewMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
    D3: DimTrait,
    T: Num,
{
    fn sub(self, lhs: Matrix<LM, D1>, rhs: Matrix<RM, D2>) {
        let mut self_ = self.into_dyn_dim();
        let lhs = lhs.into_dyn_dim();
        CopyFrom::copy_from(&mut self_, &lhs);
        let rhs = rhs * T::minus_one();
        MatrixSubAssign::sub_assign(self_, rhs.to_view());
    }
}

impl<T, LM, D1, SM, D2> MatrixSub<Matrix<LM, D1>, T> for Matrix<SM, D2>
where
    SM: ViewMutMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
    T: Num,
{
    fn sub(self, lhs: Matrix<LM, D1>, rhs: T) {
        let mut self_ = self.into_dyn_dim();
        let lhs = lhs.into_dyn_dim();
        CopyFrom::copy_from(&mut self_, &lhs);
        let rhs = rhs * T::minus_one();
        MatrixSubAssign::sub_assign(self_, rhs);
    }
}

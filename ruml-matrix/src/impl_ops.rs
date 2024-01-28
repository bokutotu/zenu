use std::ops::{Add, Mul};

use crate::{
    dim::DimTrait,
    matrix::{ToOwnedMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::ViewMemory,
    num::Num,
    operation::{add::MatrixAdd, mul::MatrixMul},
};

impl<M, D, T> Add<T> for Matrix<M, D>
where
    M: ViewMemory<Item = T>,
    D: DimTrait,
    T: Num,
{
    type Output = Matrix<M::Owned, D>;

    fn add(self, rhs: T) -> Self::Output {
        let mut owned = ToOwnedMatrix::to_owned(&self);
        let view_mut = owned.to_view_mut();
        MatrixAdd::add(view_mut, self, rhs);
        owned
    }
}

impl<M1, M2, D1, D2, T> Add<Matrix<M1, D1>> for Matrix<M2, D2>
where
    M1: ViewMemory<Item = T>,
    M2: ViewMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
    T: Num,
{
    type Output = Matrix<M2::Owned, D2>;

    fn add(self, rhs: Matrix<M1, D1>) -> Self::Output {
        let mut owned = ToOwnedMatrix::to_owned(&self);
        let view_mut = owned.to_view_mut();
        MatrixAdd::add(view_mut, self, rhs);
        owned
    }
}

impl<M, D, T> Mul<T> for Matrix<M, D>
where
    M: ViewMemory<Item = T>,
    D: DimTrait,
    T: Num,
{
    type Output = Matrix<M::Owned, D>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut owned = ToOwnedMatrix::to_owned(&self);
        let view_mut = owned.to_view_mut();
        MatrixMul::mul(view_mut, self, rhs);
        owned
    }
}

impl<M1, M2, D1, D2, T> Mul<Matrix<M1, D1>> for Matrix<M2, D2>
where
    M1: ViewMemory<Item = T>,
    M2: ViewMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
    T: Num,
{
    type Output = Matrix<M2::Owned, D2>;

    fn mul(self, rhs: Matrix<M1, D1>) -> Self::Output {
        let mut owned = ToOwnedMatrix::to_owned(&self);
        let view_mut = owned.to_view_mut();
        MatrixMul::mul(view_mut, self, rhs);
        owned
    }
}

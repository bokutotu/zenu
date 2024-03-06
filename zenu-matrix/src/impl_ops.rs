use std::ops::{Add, Div, Mul, Sub};

use crate::{
    dim::DimTrait,
    matrix::{ToOwnedMatrix, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::{ToOwnedMemory, ToViewMemory, View},
    memory_impl::{OwnedMem, ViewMem},
    num::Num,
    operation::{
        add::{MatrixAdd, MatrixAddAssign},
        div::MatrixDivAssign,
        mul::MatrixMul,
        sub::{MatrixSub, MatrixSubAssign},
    },
};

impl<'a, D, T> Add<T> for Matrix<ViewMem<'a, T>, D>
where
    D: DimTrait,
    T: Num,
{
    type Output = Matrix<OwnedMem<T>, D>;

    fn add(self, rhs: T) -> Self::Output {
        let mut owned = ToOwnedMatrix::to_owned(&self);
        MatrixAddAssign::add_assign(&mut owned, self, rhs);
        owned
    }
}

impl<D1, D2, M1, M2, T> Add<Matrix<M1, D1>> for Matrix<M2, D2>
where
    D1: DimTrait,
    D2: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMemory<Item = T> + ToOwnedMemory,
    T: Num,
{
    type Output = Matrix<M2::Owned, D2>;

    fn add(self, rhs: Matrix<M1, D1>) -> Self::Output {
        MatrixAdd::add(self, rhs)
    }
}

impl<M, D, T> Mul<T> for Matrix<M, D>
where
    M: View<Item = T>,
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

impl<'a, 'b, D1, D2, T> Mul<Matrix<ViewMem<'a, T>, D1>> for Matrix<ViewMem<'b, T>, D2>
where
    D1: DimTrait,
    D2: DimTrait,
    T: Num,
{
    type Output = Matrix<OwnedMem<T>, D2>;

    fn mul(self, rhs: Matrix<ViewMem<T>, D1>) -> Self::Output {
        let mut owned = ToOwnedMatrix::to_owned(&self);
        let view_mut = owned.to_view_mut();
        MatrixMul::mul(view_mut, self, rhs);
        owned
    }
}

impl<T, D, M> Sub<T> for Matrix<M, D>
where
    M: ToViewMemory<Item = T> + ToOwnedMemory,
    D: DimTrait,
    T: Num,
{
    type Output = Matrix<M::Owned, D>;

    fn sub(self, rhs: T) -> Self::Output {
        MatrixSub::sub(self, rhs)
    }
}

impl<T, D1, D2, M1, M2> Sub<Matrix<M1, D1>> for Matrix<M2, D2>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMemory<Item = T> + ToOwnedMemory,
{
    type Output = Matrix<M2::Owned, D2>;

    fn sub(self, rhs: Matrix<M1, D1>) -> Self::Output {
        MatrixSub::sub(self, rhs)
    }
}

impl<M1, M2, D1, D2, T> Div<Matrix<M1, D1>> for Matrix<M2, D2>
where
    D1: DimTrait,
    D2: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMemory<Item = T> + ToOwnedMemory,
    T: Num,
{
    type Output = Matrix<M2::Owned, D2>;

    fn div(self, rhs: Matrix<M1, D1>) -> Self::Output {
        let mut owned = ToOwnedMatrix::to_owned(&self);
        MatrixDivAssign::div_assign(&mut owned, rhs);
        owned
    }
}

impl<M, D, T> Div<T> for Matrix<M, D>
where
    M: ToViewMemory<Item = T> + ToOwnedMemory,
    D: DimTrait,
    T: Num,
{
    type Output = Matrix<M::Owned, D>;

    fn div(self, rhs: T) -> Self::Output {
        let mut owned = ToOwnedMatrix::to_owned(&self);
        MatrixDivAssign::div_assign(&mut owned, rhs);
        owned
    }
}

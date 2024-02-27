use std::ops::{Add, Mul, Sub};

use crate::{
    dim::DimTrait,
    matrix::{ToOwnedMatrix, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::View,
    memory_impl::{OwnedMem, ViewMem},
    num::Num,
    operation::{
        add::{MatrixAdd, MatrixAddAssign},
        mul::MatrixMul,
        sub::MatrixSubAssign,
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
        let view_mut = owned.to_view_mut();
        MatrixAdd::add(view_mut, self, rhs);
        owned
    }
}

impl<'a, 'b, D1, D2, T> Add<Matrix<ViewMem<'a, T>, D1>> for Matrix<ViewMem<'b, T>, D2>
where
    D1: DimTrait,
    D2: DimTrait,
    T: Num,
{
    type Output = Matrix<OwnedMem<T>, D2>;

    fn add(self, rhs: Matrix<ViewMem<T>, D1>) -> Self::Output {
        let mut owned = ToOwnedMatrix::to_owned(&self);
        MatrixAddAssign::add_assign(owned.to_view_mut(), rhs);
        owned
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

impl<'a, D1, T> Sub<T> for Matrix<ViewMem<'a, T>, D1>
where
    D1: DimTrait,
    T: Num,
{
    type Output = Matrix<OwnedMem<T>, D1>;

    fn sub(self, rhs: T) -> Self::Output {
        let mut ans = ToOwnedMatrix::to_owned(&self);
        ans.to_view_mut().sub_assign(rhs);
        ans
    }
}

impl<'a, 'b, D1, D2, T> Sub<Matrix<ViewMem<'a, T>, D1>> for Matrix<ViewMem<'b, T>, D2>
where
    D1: DimTrait,
    D2: DimTrait,
    T: Num,
{
    type Output = Matrix<OwnedMem<T>, D2>;

    fn sub(self, rhs: Matrix<ViewMem<T>, D1>) -> Self::Output {
        let mut ans = ToOwnedMatrix::to_owned(&self);
        ans.to_view_mut().sub_assign(rhs.to_view());
        ans
    }
}

use crate::{
    dim::DimTrait,
    matrix::{OwnedMatrix, ToViewMatrix},
    matrix_impl::Matrix,
    memory::{ToOwnedMemory, ToViewMemory, ToViewMutMemory},
    num::Num,
};

use super::add::{MatrixAdd, MatrixAddAssign};

pub trait MatrixSub<L> {
    type Output: OwnedMatrix;
    fn sub(self, lhs: L) -> Self::Output;
}

pub trait MatrixSubAssign<L, R> {
    fn sub_assign(&mut self, lhs: L, rhs: R);
}

impl<T: Num, M: ToViewMemory + ToOwnedMemory<Item = T>, D: DimTrait> MatrixSub<T> for Matrix<M, D> {
    type Output = Matrix<M::Owned, D>;

    fn sub(self, lhs: T) -> Self::Output {
        let lhs = lhs * T::minus_one();
        MatrixAdd::add(self, lhs)
    }
}

impl<T, M1, M2, D1, D2> MatrixSub<Matrix<M1, D1>> for Matrix<M2, D2>
where
    T: Num,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMemory<Item = T> + ToOwnedMemory,
    D1: DimTrait,
    D2: DimTrait,
{
    type Output = Matrix<M2::Owned, D2>;
    fn sub(self, lhs: Matrix<M1, D1>) -> Self::Output {
        let lhs = lhs.to_view() * T::minus_one();
        MatrixAdd::add(self, lhs)
    }
}

impl<T, D1, D2, M1, M2> MatrixSubAssign<Matrix<M1, D1>, T> for Matrix<M2, D2>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMutMemory<Item = T>,
{
    fn sub_assign(&mut self, lhs: Matrix<M1, D1>, rhs: T) {
        let rhs = rhs * T::minus_one();
        MatrixAddAssign::add_assign(self, lhs, rhs);
    }
}

impl<T, D1, D2, D3, M1, M2, M3> MatrixSubAssign<Matrix<M1, D1>, Matrix<M2, D2>> for Matrix<M3, D3>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    D3: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMemory<Item = T>,
    M3: ToViewMutMemory<Item = T>,
{
    fn sub_assign(&mut self, lhs: Matrix<M1, D1>, rhs: Matrix<M2, D2>) {
        let rhs = rhs.to_view() * T::minus_one();
        MatrixAddAssign::add_assign(self, lhs, rhs);
    }
}

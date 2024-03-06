use std::ops::{Add, Div, Mul, Sub};

use crate::{
    dim::DimTrait,
    matrix::ToOwnedMatrix,
    matrix_impl::Matrix,
    memory::{ToOwnedMemory, ToViewMemory},
    num::Num,
    operation::basic_operations::{MatrixAdd, MatrixDiv, MatrixMul, MatrixSub},
};

macro_rules! impl_ops {
    ($trait:ident, $method:ident, $use_trait:ident, $use_trait_method:ident) => {
        impl<T: Num, M: ToViewMemory<Item = T> + ToOwnedMemory, D: DimTrait> $trait<T>
            for Matrix<M, D>
        {
            type Output = Matrix<M::Owned, D>;

            fn $method(self, rhs: T) -> Self::Output {
                let mut owned = ToOwnedMatrix::to_owned(&self);
                $use_trait_method::$method(&mut owned, self, rhs);
                owned
            }
        }

        impl<
                T: Num,
                M1: ToViewMemory<Item = T>,
                M2: ToOwnedMemory<Item = T> + ToViewMemory<Item = T>,
                D1: DimTrait,
                D2: DimTrait,
            > $trait<Matrix<M1, D1>> for Matrix<M2, D2>
        {
            type Output = Matrix<M2::Owned, D2>;

            fn $method(self, rhs: Matrix<M1, D1>) -> Self::Output {
                let mut owned = ToOwnedMatrix::to_owned(&self);
                $use_trait_method::$method(&mut owned, self, rhs);
                owned
            }
        }
    };
}
impl_ops!(Add, add, MatrixAdd, MatrixAdd);
impl_ops!(Mul, mul, MatrixMul, MatrixMul);
impl_ops!(Sub, sub, MatrixSub, MatrixSub);
impl_ops!(Div, div, MatrixDiv, MatrixDiv);

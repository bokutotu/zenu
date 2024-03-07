use std::ops::{Add, Div, Mul, Sub};

use crate::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToOwnedMatrix},
    matrix_impl::Matrix,
    memory::{ToOwnedMemory, ToViewMemory},
    memory_impl::OwnedMem,
    num::Num,
    operation::{
        basic_operations::{MatrixAdd, MatrixDiv, MatrixMul, MatrixSub},
        zeros::Zeros,
    },
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
                M2: ToViewMemory<Item = T>,
                D1: DimTrait,
                D2: DimTrait,
            > $trait<Matrix<M1, D1>> for Matrix<M2, D2>
        {
            type Output = Matrix<OwnedMem<T>, D2>;

            fn $method(self, rhs: Matrix<M1, D1>) -> Self::Output {
                let larger_shape = if self.shape().len() >= rhs.shape().len() {
                    DimDyn::from(self.shape().slice())
                } else {
                    DimDyn::from(rhs.shape().slice())
                };
                let mut owned = Self::Output::zeros(larger_shape.slice());
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

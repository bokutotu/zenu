use crate::{
    dim_impl::{Dim1, Dim2, Dim3, Dim4},
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::{Memory, ToOwnedMemory, ToViewMemory},
    num::Num,
    operation::{add::MatrixAdd, mul::MatrixMul, zeros::Zeros},
};

use std::ops::{Add, Mul};

macro_rules! impl_scalar {
    ($dim:ty, $trait:ident, $func:ident, $matrix_trait:ident, $matrix_fn:ident) => {
        impl<T: Num, TV: ToOwnedMemory + Memory<Item = T> + ToViewMemory> $trait<T>
            for Matrix<TV, $dim>
        {
            type Output = Matrix<<TV as ToOwnedMemory>::Owned, $dim>;

            fn $func(self, rhs: T) -> Self::Output {
                let mut ans = <Self::Output as Zeros<$dim>>::zeros(self.shape());
                $matrix_trait::$matrix_fn(&mut ans.to_view_mut(), &self.to_view(), &rhs);
                ans
            }
        }

        impl<T: Num, TV: ToOwnedMemory + Memory<Item = T> + ToViewMemory> $trait<T>
            for &Matrix<TV, $dim>
        {
            type Output = Matrix<<TV as ToOwnedMemory>::Owned, $dim>;

            fn $func(self, rhs: T) -> Self::Output {
                let mut ans = <Self::Output as Zeros<$dim>>::zeros(self.shape());
                $matrix_trait::$matrix_fn(&mut ans.to_view_mut(), &self.to_view(), &rhs);
                ans
            }
        }
    };
}
macro_rules! impl_scalars {
    ($trait:ident, $func:ident, $matrix_trait:ident, $matrix_fn:ident) => {
        impl_scalar!(Dim1, $trait, $func, $matrix_trait, $matrix_fn);
        impl_scalar!(Dim2, $trait, $func, $matrix_trait, $matrix_fn);
        impl_scalar!(Dim3, $trait, $func, $matrix_trait, $matrix_fn);
        impl_scalar!(Dim4, $trait, $func, $matrix_trait, $matrix_fn);
    };
}

macro_rules! impl_matrix_operation {
    ($dim1:ty, $dim2:ty, $trait:ident, $func:ident, $matrix_trait:ident, $matrix_fn:ident) => {
        impl<T: Num, M: ToOwnedMemory + Memory<Item = T> + ToViewMemory> $trait<Matrix<M, $dim1>>
            for Matrix<M, $dim2>
        {
            type Output = Matrix<<M as ToOwnedMemory>::Owned, $dim2>;

            fn $func(self, rhs: Matrix<M, $dim1>) -> Self::Output {
                let mut ans = <Self::Output as Zeros<$dim2>>::zeros(self.shape());
                $matrix_trait::$matrix_fn(&mut ans.to_view_mut(), &self.to_view(), &rhs.to_view());
                ans
            }
        }

        impl<T: Num, M: ToOwnedMemory + Memory<Item = T> + ToViewMemory> $trait<&Matrix<M, $dim1>>
            for Matrix<M, $dim2>
        {
            type Output = Matrix<<M as ToOwnedMemory>::Owned, $dim2>;

            fn $func(self, rhs: &Matrix<M, $dim1>) -> Self::Output {
                let mut ans = <Self::Output as Zeros<$dim2>>::zeros(self.shape());
                $matrix_trait::$matrix_fn(&mut ans.to_view_mut(), &self.to_view(), &rhs.to_view());
                ans
            }
        }

        impl<T: Num, M: ToOwnedMemory + Memory<Item = T> + ToViewMemory> $trait<&Matrix<M, $dim1>>
            for &Matrix<M, $dim2>
        {
            type Output = Matrix<<M as ToOwnedMemory>::Owned, $dim2>;

            fn $func(self, rhs: &Matrix<M, $dim1>) -> Self::Output {
                let mut ans = <Self::Output as Zeros<$dim2>>::zeros(self.shape());
                $matrix_trait::$matrix_fn(&mut ans.to_view_mut(), &self.to_view(), &rhs.to_view());
                ans
            }
        }

        impl<T: Num, M: ToOwnedMemory + Memory<Item = T> + ToViewMemory> $trait<Matrix<M, $dim1>>
            for &Matrix<M, $dim2>
        {
            type Output = Matrix<<M as ToOwnedMemory>::Owned, $dim2>;

            fn $func(self, rhs: Matrix<M, $dim1>) -> Self::Output {
                let mut ans = <Self::Output as Zeros<$dim2>>::zeros(self.shape());
                $matrix_trait::$matrix_fn(&mut ans.to_view_mut(), &self.to_view(), &rhs.to_view());
                ans
            }
        }
    };
}
macro_rules! impl_matrix_matrix_ops {
    ($trait:ident, $func:ident, $matrix_trait:ident, $matrix_fn:ident) => {
        impl_matrix_operation!(Dim1, Dim1, $trait, $func, $matrix_trait, $matrix_fn);
        impl_matrix_operation!(Dim1, Dim2, $trait, $func, $matrix_trait, $matrix_fn);
        impl_matrix_operation!(Dim1, Dim3, $trait, $func, $matrix_trait, $matrix_fn);
        impl_matrix_operation!(Dim1, Dim4, $trait, $func, $matrix_trait, $matrix_fn);
        impl_matrix_operation!(Dim2, Dim2, $trait, $func, $matrix_trait, $matrix_fn);
        impl_matrix_operation!(Dim2, Dim3, $trait, $func, $matrix_trait, $matrix_fn);
        impl_matrix_operation!(Dim2, Dim4, $trait, $func, $matrix_trait, $matrix_fn);
        impl_matrix_operation!(Dim3, Dim3, $trait, $func, $matrix_trait, $matrix_fn);
        impl_matrix_operation!(Dim3, Dim4, $trait, $func, $matrix_trait, $matrix_fn);
        impl_matrix_operation!(Dim4, Dim4, $trait, $func, $matrix_trait, $matrix_fn);
    };
}

macro_rules! impl_ops {
    ($trait:ident, $func:ident, $matrix_trait:ident, $matrix_fn:ident) => {
        impl_scalars!($trait, $func, $matrix_trait, $matrix_fn);
        impl_matrix_matrix_ops!($trait, $func, $matrix_trait, $matrix_fn);
    };
}

impl_ops!(Add, add, MatrixAdd, add);
impl_ops!(Mul, mul, MatrixMul, mul);

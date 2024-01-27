use std::ops::Add;

use crate::{
    dim::DimTrait,
    matrix::{MatrixBase, OwnedMatrix},
    num::Num,
    operation::zeros::Zeros,
};

pub trait Ones: MatrixBase {
    fn ones(dim: Self::Dim) -> Self;
}

impl<D, T, OM> Ones for OM
where
    D: DimTrait,
    T: Num,
    OM: OwnedMatrix + MatrixBase<Dim = D, Item = T> + Add<T, Output = OM>,
{
    fn ones(dim: D) -> Self {
        let mut m = Self::zeros(dim);
        m = m + T::one();
        m
    }
}

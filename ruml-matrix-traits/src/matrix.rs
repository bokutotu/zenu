use std::ops::{Index, IndexMut};

use crate::{
    dim::DimTrait,
    index::{IndexTrait, ShapeStride},
    memory::Memory,
};

pub trait Matrix<IT>: Index<IT> + IndexMut<IT> + Clone
where
    IT: IndexTrait,
{
    type Dim: DimTrait;
    type Memory: Memory;

    fn shape_stride(&self) -> ShapeStride<Self::Dim>;
    fn shape(&self) -> Self::Dim;
    fn stride(&self) -> Self::Dim;
    fn memory(&self) -> &Self::Memory;
}

pub trait OwnedMatrix<IT>: Matrix<IT>
where
    IT: IndexTrait,
{
    type View: ViewMatrix<IT>;

    fn into_view(self) -> Self::View;
}

pub trait ViewMatrix<IT>: Matrix<IT>
where
    IT: IndexTrait,
{
    type Owned: OwnedMatrix<IT>;

    fn into_owned(&self) -> Self::Owned;
}

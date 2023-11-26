use crate::{
    dim::DimTrait,
    index::{ShapeStride, SliceTrait},
    memory::Memory,
};

pub trait Matrix: Clone {
    type Dim: DimTrait;
    type Memory: Memory;

    fn shape_stride(&self) -> ShapeStride<Self::Dim>;
    fn memory(&self) -> &Self::Memory;
}

pub trait OwnedMatrix: Matrix {
    type View<'a>: ViewMatrix
    where
        Self: 'a;

    fn to_view(&self) -> Self::View<'_>;
    fn construct(data: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self;
    fn from_vec(vec: Vec<<<Self as Matrix>::Memory as Memory>::Item>, dim: Self::Dim) -> Self;
}

pub trait ViewMatrix: Matrix {
    type Owned: OwnedMatrix;

    fn construct(data: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self;
    fn to_owned(&self) -> Self::Owned;
}

pub trait MatrixSlice<S: SliceTrait>: Matrix {
    type Output<'a>: ViewMatrix
    where
        Self: 'a;
    fn slice(&self, index: S) -> Self::Output<'_>;
}

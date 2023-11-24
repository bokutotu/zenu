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
    fn from_vec(vec: Vec<<<Self as Matrix>::Memory as Memory>::Item>, dim: Self::Dim) -> Self;
}

pub trait OwnedMatrix: Matrix {
    type View: ViewMatrix;

    fn to_view(&self) -> Self::View;
    fn construct(data: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self;
}

pub trait ViewMatrix: Matrix {
    type Owned: OwnedMatrix;

    fn construct(data: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self;
    fn to_owned(&self) -> Self::Owned;
}

pub trait MatrixSlice<S: SliceTrait>: Matrix {
    type Output: ViewMatrix;
    fn slice(&self, index: S) -> Self::Output;
}

use crate::{
    dim::DimTrait,
    index::{IndexAxisTrait, ShapeStride, SliceTrait},
    memory::Memory,
};

pub trait Matrix: Clone {
    type Dim: DimTrait;
    type Memory: Memory;

    fn shape_stride(&self) -> ShapeStride<Self::Dim>;
    fn memory(&self) -> &Self::Memory;
    fn construct(data: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self;
}

pub trait OwnedMatrix<'a>: Matrix {
    type View: ViewMatrix<'a>;

    fn to_view(&'a self) -> Self::View;
    fn from_vec(vec: Vec<<<Self as Matrix>::Memory as Memory>::Item>, dim: Self::Dim) -> Self;
}

pub trait ViewMatrix<'a>: Matrix {
    type Owned: OwnedMatrix<'a>;

    fn to_owned(&self) -> Self::Owned;
}

pub trait MatrixSlice<'a, S: SliceTrait>: Matrix {
    type Output: ViewMatrix<'a>;

    fn slice(&'a self, index: S) -> Self::Output;
}

pub trait IndexAxis<I: IndexAxisTrait> {
    type Output<'a>: ViewMatrix<'a>
    where
        Self: 'a;

    fn index_axis(&self, index: I) -> Self::Output<'_>;
}

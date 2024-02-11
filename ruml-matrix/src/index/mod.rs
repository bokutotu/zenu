pub mod index_impl;

pub use index_impl::{Index0D, Index1D, Index2D, Index3D};

use crate::{dim::DimTrait, shape_stride::ShapeStride};

pub trait SliceTrait {
    type Dim: DimTrait;
    fn sliced_shape_stride(&self, shape: Self::Dim, stride: Self::Dim) -> ShapeStride<Self::Dim>;
    fn sliced_offset(&self, stride: Self::Dim, original_offset: usize) -> usize;
}

pub trait IndexAxisTrait {
    fn get_shape_stride<Din: DimTrait, Dout: DimTrait>(
        &self,
        shape: Din,
        stride: Din,
    ) -> ShapeStride<Dout>;
    fn offset<Din: DimTrait>(&self, stride: Din) -> usize;
}

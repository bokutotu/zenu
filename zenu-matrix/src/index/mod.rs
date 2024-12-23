#![expect(clippy::module_name_repetitions)]

pub mod index_dyn_impl;
pub mod index_impl;

pub use index_impl::{Index0D, Index1D, Index2D, Index3D};

use crate::{dim::DimTrait, shape_stride::ShapeStride};

pub trait SliceTrait: Copy {
    type Dim: DimTrait;
    fn sliced_shape_stride(&self, shape: Self::Dim, stride: Self::Dim) -> ShapeStride<Self::Dim>;
    fn sliced_offset(&self, stride: Self::Dim) -> usize;
}

pub trait IndexAxisTrait: Copy {
    fn get_shape_stride<Din: DimTrait, Dout: DimTrait>(
        &self,
        shape: Din,
        stride: Din,
    ) -> ShapeStride<Dout>;
    fn offset<Din: DimTrait>(&self, stride: Din) -> usize;
}

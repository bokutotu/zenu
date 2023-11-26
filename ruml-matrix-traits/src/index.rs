use crate::dim::DimTrait;

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct ShapeStride<D: DimTrait> {
    shape: D,
    stride: D,
}

impl<D: DimTrait> ShapeStride<D> {
    pub fn new(shape: D, stride: D) -> Self {
        Self { shape, stride }
    }

    pub fn shape(&self) -> D {
        self.shape
    }

    pub fn stride(&self) -> D {
        self.stride
    }
}

pub trait SliceTrait {
    type Dim: DimTrait;
    fn sliced_shape_stride(&self, shape: &Self::Dim, stride: &Self::Dim) -> ShapeStride<Self::Dim>;
    fn sliced_offset(&self, stride: &Self::Dim, original_offset: usize) -> usize;
}

/// Matrixに対して、Indexを取得してTを取得するのに使用するトレイト
pub trait IndexTrait {
    type Dim: DimTrait;
    fn offset(&self, shape: &Self::Dim, stride: &Self::Dim) -> usize;
}

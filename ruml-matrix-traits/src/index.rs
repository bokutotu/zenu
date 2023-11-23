use crate::dim::DimTrait;

pub struct ShapeStride<D: DimTrait> {
    shape: D,
    stride: D,
}

impl<D: DimTrait> ShapeStride<D> {
    pub fn new(shape: D, stride: D) -> Self {
        Self { shape, stride }
    }

    pub fn shape(&self) -> D {
        *&self.shape
    }

    pub fn stride(&self) -> D {
        *&self.stride
    }
}

pub trait IndexTrait {
    type InDim: DimTrait;
    type OutDim: DimTrait;
    fn sliced_shape_stride(
        &self,
        shape: &Self::InDim,
        stride: &Self::InDim,
    ) -> ShapeStride<Self::OutDim>;
}

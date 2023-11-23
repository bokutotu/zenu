use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use ruml_dim_impl::{Dim0, Dim1, Dim2, Dim3, Dim4};
use ruml_matrix_traits::index::{IndexTrait, ShapeStride};

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct SliceDim {
    pub(crate) start: Option<usize>,
    pub(crate) end: Option<usize>,
    pub(crate) step: Option<usize>,
}

impl SliceDim {
    pub fn step(self, step: usize) -> Self {
        Self {
            start: self.start,
            end: self.end,
            step: Some(step),
        }
    }

    fn validate(&self, dim: usize) -> bool {
        let start = self.start.unwrap_or(0);
        let end = self.end.unwrap_or(dim - 1);
        let step = self.step.unwrap_or(1);

        if start > end {
            return false;
        }

        if start > dim {
            return false;
        }

        if end > dim {
            return false;
        }

        if step == 0 {
            return false;
        }

        true
    }

    fn new_dim_unchanged(&self, dim: usize) -> usize {
        let start = self.start.unwrap_or(0);
        let end = self.end.unwrap_or(dim - 1);
        let step = self.step.unwrap_or(1);

        if step == 1 {
            return end - start + 1;
        }

        let mut new_dim = 0;
        let mut i = start;
        while i <= end {
            new_dim += 1;
            i += step;
        }
        new_dim
    }

    fn new_dim(&self, dim: usize) -> usize {
        if self.validate(dim) {
            return self.new_dim_unchanged(dim);
        }
        panic!("invalid slice");
    }

    fn new_stride(&self, stride: usize) -> usize {
        let step = self.step.unwrap_or(1);
        stride * step
    }
}

impl From<Range<usize>> for SliceDim {
    fn from(range: Range<usize>) -> Self {
        SliceDim {
            start: Some(range.start),
            end: Some(range.end),
            step: None,
        }
    }
}

impl From<RangeFull> for SliceDim {
    fn from(_: RangeFull) -> Self {
        SliceDim {
            start: None,
            end: None,
            step: None,
        }
    }
}

impl From<RangeTo<usize>> for SliceDim {
    fn from(range: RangeTo<usize>) -> Self {
        SliceDim {
            start: None,
            end: Some(range.end),
            step: None,
        }
    }
}

impl From<RangeFrom<usize>> for SliceDim {
    fn from(range: RangeFrom<usize>) -> Self {
        SliceDim {
            start: Some(range.start),
            end: None,
            step: None,
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct Slice0D {}

impl IndexTrait for Slice0D {
    type InDim = Dim0;
    type OutDim = Dim0;

    fn sliced_shape_stride(
        &self,
        shape: &Self::InDim,
        stride: &Self::InDim,
    ) -> ShapeStride<Self::OutDim> {
        ShapeStride::new(*shape, *stride)
    }
}

// IndexND 構造体の定義
#[derive(Clone, Debug, Copy, PartialEq)]
pub struct Slice1D {
    pub(crate) index: SliceDim,
}

impl IndexTrait for Slice1D {
    type InDim = Dim1;
    type OutDim = Dim1;

    fn sliced_shape_stride(
        &self,
        shape: &Self::InDim,
        stride: &Self::InDim,
    ) -> ShapeStride<Self::OutDim> {
        let new_dim = self.index.new_dim(shape.dim());
        let new_stride = self.index.new_stride(stride.dim());
        ShapeStride::new(Dim1::new(new_dim), Dim1::new(new_stride))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Slice2D {
    pub(crate) index: [SliceDim; 2],
}

impl IndexTrait for Slice2D {
    type InDim = Dim2;
    type OutDim = Dim2;

    fn sliced_shape_stride(
        &self,
        shape: &Self::InDim,
        stride: &Self::InDim,
    ) -> ShapeStride<Self::OutDim> {
        let new_dim0 = self.index[0].new_dim(shape[0]);
        let new_dim1 = self.index[1].new_dim(shape[1]);
        let new_stride0 = self.index[0].new_stride(stride[0]);
        let new_stride1 = self.index[1].new_stride(stride[1]);
        ShapeStride::new(
            Dim2::new([new_dim0, new_dim1]),
            Dim2::new([new_stride0, new_stride1]),
        )
    }
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct Slice3D {
    pub(crate) index: [SliceDim; 3],
}

impl IndexTrait for Slice3D {
    type InDim = Dim3;
    type OutDim = Dim3;

    fn sliced_shape_stride(
        &self,
        shape: &Self::InDim,
        stride: &Self::InDim,
    ) -> ShapeStride<Self::OutDim> {
        let new_dim0 = self.index[0].new_dim(shape[0]);
        let new_dim1 = self.index[1].new_dim(shape[1]);
        let new_dim2 = self.index[2].new_dim(shape[2]);
        let new_stride0 = self.index[0].new_stride(stride[0]);
        let new_stride1 = self.index[1].new_stride(stride[1]);
        let new_stride2 = self.index[2].new_stride(stride[2]);
        ShapeStride::new(
            Dim3::new([new_dim0, new_dim1, new_dim2]),
            Dim3::new([new_stride0, new_stride1, new_stride2]),
        )
    }
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct Slice4D {
    pub(crate) index: [SliceDim; 4],
}

impl IndexTrait for Slice4D {
    type InDim = Dim4;
    type OutDim = Dim4;

    fn sliced_shape_stride(
        &self,
        shape: &Self::InDim,
        stride: &Self::InDim,
    ) -> ShapeStride<Self::OutDim> {
        let new_dim0 = self.index[0].new_dim(shape[0]);
        let new_dim1 = self.index[1].new_dim(shape[1]);
        let new_dim2 = self.index[2].new_dim(shape[2]);
        let new_dim3 = self.index[3].new_dim(shape[3]);
        let new_stride0 = self.index[0].new_stride(stride[0]);
        let new_stride1 = self.index[1].new_stride(stride[1]);
        let new_stride2 = self.index[2].new_stride(stride[2]);
        let new_stride3 = self.index[3].new_stride(stride[3]);
        ShapeStride::new(
            Dim4::new([new_dim0, new_dim1, new_dim2, new_dim3]),
            Dim4::new([new_stride0, new_stride1, new_stride2, new_stride3]),
        )
    }
}

use crate::{dim::DimTrait, shape_stride::ShapeStride};

use super::IndexAxisTrait;

#[derive(Clone, Copy, Debug)]
pub struct Index {
    axis: usize,
    index: usize,
}

impl Index {
    #[must_use]
    pub fn new(axis: usize, index: usize) -> Self {
        Self { axis, index }
    }

    #[must_use]
    pub fn axis(&self) -> usize {
        self.axis
    }

    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }
}

impl IndexAxisTrait for Index {
    fn get_shape_stride<Din: DimTrait, Dout: DimTrait>(
        &self,
        shape: Din,
        stride: Din,
    ) -> ShapeStride<Dout> {
        let mut shape_v = Vec::new();
        let mut stride_v = Vec::new();
        for i in 0..shape.len() {
            if i == self.axis {
                continue;
            }
            shape_v.push(shape[i]);
            stride_v.push(stride[i]);
        }

        let new_shape = Dout::from(&shape_v as &[usize]);
        let new_stride = Dout::from(&stride_v as &[usize]);
        ShapeStride::new(new_shape, new_stride)
    }
    fn offset<Din: DimTrait>(&self, stride: Din) -> usize {
        stride[self.axis] * self.index
    }
}

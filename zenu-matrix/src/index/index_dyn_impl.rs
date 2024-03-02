use crate::{dim::DimTrait, shape_stride::ShapeStride};

use super::IndexAxisTrait;

pub struct Index {
    axis: usize,
    index: usize,
}

impl Index {
    pub fn new(axis: usize, index: usize) -> Self {
        Self { axis, index }
    }

    pub fn axis(&self) -> usize {
        self.axis
    }

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
            } else {
                shape_v.push(shape[i]);
                stride_v.push(stride[i]);
            }
        }

        let new_shape = Dout::from(&shape_v);
        let new_stride = Dout::from(&stride_v);
        ShapeStride::new(new_shape, new_stride)
    }
    fn offset<Din: DimTrait>(&self, stride: Din) -> usize {
        stride[self.axis] * self.index
    }
}

use super::slice_dim::SliceDim;
use crate::{
    dim_impl::DimDyn,
    index::{ShapeStride, SliceTrait},
};

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct Slice {
    pub index: [SliceDim; 4],
    pub len: usize,
}

impl SliceTrait for Slice {
    type Dim = DimDyn;

    fn sliced_shape_stride(&self, shape: Self::Dim, stride: Self::Dim) -> ShapeStride<Self::Dim> {
        let mut new_shape = shape;
        let mut new_stride = stride;

        for i in 0..self.len {
            new_shape[i] = self.index[i].new_dim(shape[i]);
            new_stride[i] = self.index[i].new_stride(stride[i]);
        }

        ShapeStride::new(new_shape, new_stride)
    }

    fn sliced_offset(&self, stride: Self::Dim, original_offset: usize) -> usize {
        let mut offset = 0;

        for i in 0..self.len {
            let start = self.index[i].start.unwrap_or(0);
            offset += start * stride[i];
        }

        offset + original_offset
    }
}

impl From<&[SliceDim]> for Slice {
    fn from(s: &[SliceDim]) -> Self {
        if s.len() > 4 {
            panic!("too many slice dimensions");
        } else if s.len() == 1 {
            Slice {
                index: [
                    s[0],
                    SliceDim::default(),
                    SliceDim::default(),
                    SliceDim::default(),
                ],
                len: 1,
            }
        } else if s.len() == 2 {
            Slice {
                index: [s[0], s[1], SliceDim::default(), SliceDim::default()],
                len: 2,
            }
        } else if s.len() == 3 {
            Slice {
                index: [s[0], s[1], s[2], SliceDim::default()],
                len: 3,
            }
        } else {
            Slice {
                index: [s[0], s[1], s[2], s[3]],
                len: 4,
            }
        }
    }
}

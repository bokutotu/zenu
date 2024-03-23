use super::slice_dim::SliceDim;
use crate::{dim::DimDyn, index::SliceTrait, shape_stride::ShapeStride};

#[derive(Clone, Debug, Copy, PartialEq)]

pub struct Slice {
    pub index: [SliceDim; 6],
    pub len: usize,
}

impl SliceTrait for Slice {
    type Dim = DimDyn;

    fn sliced_shape_stride(&self, shape: Self::Dim, stride: Self::Dim) -> ShapeStride<Self::Dim> {
        let mut new_shape = DimDyn::default();
        let mut new_stride = DimDyn::default();

        for i in 0..self.len {
            match self.index[i].new_dim(shape[i]) {
                0 => continue,
                new_dim => {
                    new_shape.push_dim(new_dim);
                    new_stride.push_dim(self.index[i].new_stride(stride[i]));
                }
            }
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
        if s.len() > 6 {
            panic!("too many slice dimensions");
        } else if s.len() == 1 {
            Slice {
                index: [
                    s[0],
                    SliceDim::default(),
                    SliceDim::default(),
                    SliceDim::default(),
                    SliceDim::default(),
                    SliceDim::default(),
                ],
                len: 1,
            }
        } else if s.len() == 2 {
            Slice {
                index: [
                    s[0],
                    s[1],
                    SliceDim::default(),
                    SliceDim::default(),
                    SliceDim::default(),
                    SliceDim::default(),
                ],
                len: 2,
            }
        } else if s.len() == 3 {
            Slice {
                index: [
                    s[0],
                    s[1],
                    s[2],
                    SliceDim::default(),
                    SliceDim::default(),
                    SliceDim::default(),
                ],
                len: 3,
            }
        } else if s.len() == 4 {
            Slice {
                index: [
                    s[0],
                    s[1],
                    s[2],
                    s[3],
                    SliceDim::default(),
                    SliceDim::default(),
                ],
                len: 4,
            }
        } else if s.len() == 5 {
            Slice {
                index: [s[0], s[1], s[2], s[3], s[4], SliceDim::default()],
                len: 5,
            }
        } else {
            Slice {
                index: [s[0], s[1], s[2], s[3], s[4], s[5]],
                len: 6,
            }
        }
    }
}

#[cfg(test)]
mod slice_dyn_slice {
    use crate::{dim::DimDyn, index::SliceTrait, slice_dynamic};

    #[test]
    fn dyn_slice() {
        let shape = DimDyn::new(&[2, 3, 4]);
        let stride = DimDyn::new(&[12, 4, 1]);
        let slice = slice_dynamic!(.., 1, 1..2);
        let shape_stride = slice.sliced_shape_stride(shape, stride);
        let result_shape = shape_stride.shape();
        let result_stride = shape_stride.stride();
        assert_eq!(result_shape, DimDyn::new(&[2, 1]));
        assert_eq!(result_stride, DimDyn::new(&[12, 1]));
    }
}

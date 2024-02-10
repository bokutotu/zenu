use crate::{
    dim_impl::{Dim0, Dim1, Dim2, Dim3, Dim4},
    index::{ShapeStride, SliceTrait},
};

use super::slice_dim::SliceDim;

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct Slice0D {}

impl SliceTrait for Slice0D {
    type Dim = Dim0;

    fn sliced_shape_stride(&self, shape: Self::Dim, stride: Self::Dim) -> ShapeStride<Self::Dim> {
        ShapeStride::new(shape, stride)
    }

    fn sliced_offset(&self, _stride: Self::Dim, _original_offset: usize) -> usize {
        0
    }
}

macro_rules! impl_slice_ty {
    ($impl_name:ident, $num_item:expr, $dim_ty:ty) => {
        #[derive(Clone, Debug, Copy, PartialEq)]
        pub struct $impl_name {
            pub index: [SliceDim; $num_item],
        }

        impl $impl_name {
            pub fn new(index: [SliceDim; $num_item]) -> Self {
                Self { index }
            }

            pub fn index(&self) -> &[SliceDim; $num_item] {
                &self.index
            }
        }

        impl SliceTrait for $impl_name {
            type Dim = $dim_ty;

            fn sliced_shape_stride(
                &self,
                shape: Self::Dim,
                stride: Self::Dim,
            ) -> ShapeStride<Self::Dim> {
                let mut new_shape = shape.clone();
                let mut new_stride = stride.clone();

                for i in 0..$num_item {
                    new_shape[i] = self.index[i].new_dim(shape[i]);
                    new_stride[i] = self.index[i].new_stride(stride[i]);
                }

                ShapeStride::new(new_shape, new_stride)
            }

            fn sliced_offset(&self, stride: Self::Dim, original_offset: usize) -> usize {
                let mut offset = 0;

                for i in 0..$num_item {
                    let start = self.index[i].start.unwrap_or(0);
                    offset += start * stride[i];
                }

                offset + original_offset
            }
        }
    };
}
impl_slice_ty!(Slice1D, 1, Dim1);
impl_slice_ty!(Slice2D, 2, Dim2);
impl_slice_ty!(Slice3D, 3, Dim3);
impl_slice_ty!(Slice4D, 4, Dim4);

#[cfg(test)]
mod static_dim_slice {
    use crate::dim_impl::{Dim1, Dim2, Dim3};
    use crate::index::SliceTrait;
    use crate::slice;

    #[test]
    fn sliced_1d() {
        let shape = Dim1::new([6]);
        let stride = Dim1::new([1]);
        let slice = slice!(..;2);

        let stride_shape = dbg!(slice.sliced_shape_stride(shape, stride));

        assert_eq!(stride_shape.shape(), Dim1::new([3]));
        assert_eq!(stride_shape.stride(), Dim1::new([2]));
    }

    #[test]
    fn test_sliced_shape_stride_2d() {
        let original_shape = Dim2::new([10, 20]);
        let original_stride = Dim2::new([1, 10]);
        let slice = crate::slice!(1..5;2, 3..10;1);
        let new = slice.sliced_shape_stride(original_shape, original_stride);

        assert_eq!(new.shape(), Dim2::new([2, 7]));
        assert_eq!(new.stride(), Dim2::new([2, 10]));
    }

    #[test]
    fn test_sliced_shape_stride_3d() {
        let original_shape = Dim3::new([10, 20, 30]);
        let original_stride = Dim3::new([1, 10, 200]);
        let slice = crate::slice!(1..5;2, 3..10;1, ..15;3);
        let new = slice.sliced_shape_stride(original_shape, original_stride);

        assert_eq!(new.shape(), Dim3::new([2, 7, 5]));
        assert_eq!(new.stride(), Dim3::new([2, 10, 600]),);
    }

    #[test]
    fn test_sliced_offset_2d() {
        let stride = Dim2::new([10, 1]);
        let slice = crate::slice!(1..5;2, 3..10;1);
        let offset = slice.sliced_offset(stride, 0);

        assert_eq!(offset, 13);
    }
}

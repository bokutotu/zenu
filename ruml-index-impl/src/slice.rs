// use std::ops::{Range, RangeFrom, RangeFull, RangeTo};
//
// use ruml_dim_impl::{Dim0, Dim1, Dim2, Dim3, Dim4};
// use ruml_matrix::index::{ShapeStride, SliceTrait};
//
// #[allow(unused_imports)]
// use crate::slice;
//
// #[derive(Clone, Debug, Copy, PartialEq)]
// pub struct SliceDim {
//     pub(crate) start: Option<usize>,
//     pub(crate) end: Option<usize>,
//     pub(crate) step: Option<usize>,
// }
//
// impl SliceDim {
//     pub fn step(self, step: usize) -> Self {
//         Self {
//             start: self.start,
//             end: self.end,
//             step: Some(step),
//         }
//     }
//
//     fn validate(&self, dim: usize) -> bool {
//         let start = self.start.unwrap_or(0);
//         let end = self.end.unwrap_or(dim - 1);
//         let step = self.step.unwrap_or(1);
//
//         if start > end {
//             return false;
//         }
//
//         if start > dim {
//             return false;
//         }
//
//         if end > dim {
//             return false;
//         }
//
//         if step == 0 {
//             return false;
//         }
//
//         true
//     }
//
//     fn new_dim_unchanged(&self, dim: usize) -> usize {
//         let start = self.start.unwrap_or(0);
//         let mut end = self.end.unwrap_or(dim);
//         let step = self.step.unwrap_or(1);
//
//         if end > dim {
//             end = dim;
//         }
//
//         (end - start + step - 1) / step
//     }
//
//     fn new_dim(&self, dim: usize) -> usize {
//         if self.validate(dim) {
//             return self.new_dim_unchanged(dim);
//         }
//         panic!("invalid slice");
//     }
//
//     fn new_stride(&self, stride: usize) -> usize {
//         let step = self.step.unwrap_or(1);
//         stride * step
//     }
// }
//
// impl From<Range<usize>> for SliceDim {
//     fn from(range: Range<usize>) -> Self {
//         SliceDim {
//             start: Some(range.start),
//             end: Some(range.end),
//             step: None,
//         }
//     }
// }
//
// impl From<RangeFull> for SliceDim {
//     fn from(_: RangeFull) -> Self {
//         SliceDim {
//             start: None,
//             end: None,
//             step: None,
//         }
//     }
// }
//
// impl From<RangeTo<usize>> for SliceDim {
//     fn from(range: RangeTo<usize>) -> Self {
//         SliceDim {
//             start: None,
//             end: Some(range.end),
//             step: None,
//         }
//     }
// }
//
// impl From<RangeFrom<usize>> for SliceDim {
//     fn from(range: RangeFrom<usize>) -> Self {
//         SliceDim {
//             start: Some(range.start),
//             end: None,
//             step: None,
//         }
//     }
// }
//
// #[derive(Clone, Debug, Copy, PartialEq)]
// pub struct Slice0D {}
//
// impl SliceTrait for Slice0D {
//     type Dim = Dim0;
//
//     fn sliced_shape_stride(&self, shape: &Self::Dim, stride: &Self::Dim) -> ShapeStride<Self::Dim> {
//         ShapeStride::new(*shape, *stride)
//     }
//
//     fn sliced_offset(&self, _stride: &Self::Dim, _original_offset: usize) -> usize {
//         0
//     }
// }
//
// macro_rules! impl_slice_ty {
//     ($impl_name:ident, $num_item:expr, $dim_ty:ty) => {
//         #[derive(Clone, Debug, Copy, PartialEq)]
//         pub struct $impl_name {
//             pub index: [SliceDim; $num_item],
//         }
//
//         impl $impl_name {
//             pub fn new(index: [SliceDim; $num_item]) -> Self {
//                 Self { index }
//             }
//
//             pub fn index(&self) -> &[SliceDim; $num_item] {
//                 &self.index
//             }
//         }
//
//         impl SliceTrait for $impl_name {
//             type Dim = $dim_ty;
//
//             fn sliced_shape_stride(
//                 &self,
//                 shape: &Self::Dim,
//                 stride: &Self::Dim,
//             ) -> ShapeStride<Self::Dim> {
//                 let mut new_shape = shape.clone();
//                 let mut new_stride = stride.clone();
//
//                 for i in 0..$num_item {
//                     new_shape[i] = self.index[i].new_dim(shape[i]);
//                     new_stride[i] = self.index[i].new_stride(stride[i]);
//                 }
//
//                 ShapeStride::new(new_shape, new_stride)
//             }
//
//             fn sliced_offset(&self, stride: &Self::Dim, original_offset: usize) -> usize {
//                 let mut offset = 0;
//
//                 for i in 0..$num_item {
//                     let start = self.index[i].start.unwrap_or(0);
//                     offset += start * stride[i];
//                 }
//
//                 offset + original_offset
//             }
//         }
//     };
// }
// impl_slice_ty!(Slice1D, 1, Dim1);
// impl_slice_ty!(Slice2D, 2, Dim2);
// impl_slice_ty!(Slice3D, 3, Dim3);
// impl_slice_ty!(Slice4D, 4, Dim4);
//
// #[test]
// fn slice_index() {
//     let slice_dim = SliceDim {
//         start: Some(0),
//         end: Some(10),
//         step: None,
//     };
//
//     let dim = 20;
//     let new_dim = slice_dim.new_dim(dim);
//     assert_eq!(new_dim, 10);
//     let new_stride = slice_dim.new_stride(1);
//     assert_eq!(new_stride, 1);
// }
//
// #[test]
// fn slice_index_with_stride() {
//     let slice_dim = SliceDim {
//         start: Some(0),
//         end: Some(10),
//         step: Some(2),
//     };
//
//     let dim = 20;
//     let new_dim = slice_dim.new_dim(dim);
//     assert_eq!(new_dim, 5);
//     let new_stride = slice_dim.new_stride(1);
//     assert_eq!(new_stride, 2);
// }
//
// #[test]
// fn slice_dim_full_range() {
//     let slice_dim = SliceDim {
//         start: None,
//         end: None,
//         step: None,
//     };
//
//     let dim = 20;
//     let new_dim = slice_dim.new_dim(dim);
//     assert_eq!(new_dim, 20);
//     let new_stride = slice_dim.new_stride(1);
//     assert_eq!(new_stride, 1);
// }
//
// #[test]
// fn sliced_1d() {
//     let shape = Dim1::new([6]);
//     let stride = Dim1::new([1]);
//     let slice = slice!(..;2);
//
//     let stride_shape = dbg!(slice.sliced_shape_stride(&shape, &stride));
//
//     assert_eq!(stride_shape.shape(), Dim1::new([3]));
//     assert_eq!(stride_shape.stride(), Dim1::new([2]));
// }
//
// #[test]
// fn test_sliced_shape_stride_2d() {
//     let original_shape = Dim2::new([10, 20]);
//     let original_stride = Dim2::new([1, 10]);
//     let slice = crate::slice!(1..5;2, 3..10;1);
//     let new = slice.sliced_shape_stride(&original_shape, &original_stride);
//
//     assert_eq!(new.shape(), Dim2::new([2, 7]));
//     assert_eq!(new.stride(), Dim2::new([2, 10]));
// }
//
// #[test]
// fn test_sliced_shape_stride_3d() {
//     let original_shape = Dim3::new([10, 20, 30]);
//     let original_stride = Dim3::new([1, 10, 200]);
//     let slice = crate::slice!(1..5;2, 3..10;1, ..15;3);
//     let new = slice.sliced_shape_stride(&original_shape, &original_stride);
//
//     assert_eq!(new.shape(), Dim3::new([2, 7, 5]));
//     assert_eq!(new.stride(), Dim3::new([2, 10, 600]),);
// }
//
// #[test]
// fn test_sliced_offset_2d() {
//     let stride = Dim2::new([10, 1]);
//     let slice = crate::slice!(1..5;2, 3..10;1);
//     let offset = slice.sliced_offset(&stride, 0);
//
//     assert_eq!(offset, 13);
// }

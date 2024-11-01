use crate::dim::DimTrait;
use crate::index::{IndexAxisTrait, ShapeStride};

macro_rules! impl_index_axis {
    ($impl_name:ident, $target_dim:expr) => {
        #[derive(Copy, Clone, Debug, PartialEq)]
        pub struct $impl_name(pub usize);

        impl $impl_name {
            #[must_use]
            pub fn new(index: usize) -> Self {
                $impl_name(index)
            }

            #[must_use]
            pub fn index(&self) -> usize {
                self.0
            }

            #[must_use]
            pub fn target_dim(&self) -> usize {
                $target_dim
            }

            pub fn get_shape_stride<Din: DimTrait, Dout: DimTrait>(
                &self,
                shape: &Din,
                stride: &Din,
            ) -> ShapeStride<Dout> {
                let mut shape_v = Vec::new();
                let mut stride_v = Vec::new();
                for i in 0..shape.len() {
                    if i == $target_dim {
                        continue;
                    }
                    shape_v.push(shape[i]);
                    stride_v.push(stride[i]);
                }

                let new_shape = Dout::from(&shape_v as &[usize]);
                let new_stride = Dout::from(&stride_v as &[usize]);
                ShapeStride::new(new_shape, new_stride)
            }

            pub fn get_offset<D: DimTrait>(&self, stride: D) -> usize {
                stride[$target_dim] * self.0
            }
        }
    };
}
impl_index_axis!(Index0D, 0);
impl_index_axis!(Index1D, 1);
impl_index_axis!(Index2D, 2);
impl_index_axis!(Index3D, 3);

macro_rules! impl_index_axis_trait {
    ($impl_trait:ident) => {
        impl IndexAxisTrait for $impl_trait {
            fn get_shape_stride<Din: DimTrait, Dout: DimTrait>(
                &self,
                shape: Din,
                stride: Din,
            ) -> ShapeStride<Dout> {
                self.get_shape_stride::<Din, Dout>(&shape, &stride)
            }

            fn offset<Din: DimTrait>(&self, stride: Din) -> usize {
                self.get_offset::<Din>(stride.clone())
            }
        }
    };
}
impl_index_axis_trait!(Index0D);
impl_index_axis_trait!(Index1D);
impl_index_axis_trait!(Index2D);
impl_index_axis_trait!(Index3D);

#[cfg(test)]
mod index_xd {
    use super::{Index0D, Index1D, Index2D, Index3D};
    use crate::dim::{Dim1, Dim2, Dim3, Dim4};

    #[test]
    fn offset_1d() {
        let stride = Dim1::new([1]);
        let index = Index0D::new(1);
        let offset = index.get_offset(stride);
        assert_eq!(offset, 1);
    }

    #[test]
    fn offset_2d() {
        let stride = Dim2::new([4, 1]);
        let index = Index0D::new(2);
        let offset = index.get_offset(stride);
        assert_eq!(offset, 8);
    }

    #[test]
    fn offset_3d() {
        let stride = Dim3::new([20, 5, 1]);
        let index = Index0D::new(2);
        let offset = index.get_offset(stride);
        assert_eq!(offset, 40);
    }

    #[test]
    fn shape_stride_2d_index_0() {
        let shape = Dim2::new([3, 4]);
        let stride = Dim2::new([4, 1]);

        let index = Index0D::new(1);

        let shape_stride = index.get_shape_stride::<Dim2, Dim1>(&shape, &stride);

        assert_eq!(shape_stride.shape(), Dim1::new([4]));
        assert_eq!(shape_stride.stride(), Dim1::new([1]));
    }

    #[test]
    fn shape_stride_2d_index_1() {
        let shape = Dim2::new([3, 4]);
        let stride = Dim2::new([4, 1]);

        let index = Index1D::new(1);

        let shape_stride = index.get_shape_stride::<Dim2, Dim1>(&shape, &stride);

        assert_eq!(shape_stride.shape(), Dim1::new([3]));
        assert_eq!(shape_stride.stride(), Dim1::new([4]));
    }

    #[test]
    fn shape_stride_3d_index_0() {
        let shape = Dim3::new([3, 4, 5]);
        let stride = Dim3::new([20, 5, 1]);

        let index = Index0D::new(1);

        let shape_stride = index.get_shape_stride::<Dim3, Dim2>(&shape, &stride);

        assert_eq!(shape_stride.shape(), Dim2::new([4, 5]));
        assert_eq!(shape_stride.stride(), Dim2::new([5, 1]));
    }

    #[test]
    fn shape_stride_3d_index_1() {
        let shape = Dim3::new([3, 4, 5]);
        let stride = Dim3::new([20, 5, 1]);

        let index = Index1D::new(1);

        let shape_stride = index.get_shape_stride::<Dim3, Dim2>(&shape, &stride);

        assert_eq!(shape_stride.shape(), Dim2::new([3, 5]));
        assert_eq!(shape_stride.stride(), Dim2::new([20, 1]));
    }

    #[test]
    fn shape_stride_3d_index_2() {
        let shape = Dim3::new([3, 4, 5]);
        let stride = Dim3::new([20, 5, 1]);

        let index = Index2D::new(1);

        let shape_stride = index.get_shape_stride::<Dim3, Dim2>(&shape, &stride);

        assert_eq!(shape_stride.shape(), Dim2::new([3, 4]));
        assert_eq!(shape_stride.stride(), Dim2::new([20, 5]));
    }

    #[test]
    fn shape_stride_4d_index_0() {
        let shape = Dim4::new([3, 4, 5, 6]);
        let stride = Dim4::new([120, 30, 6, 1]);

        let index = Index0D::new(1);

        let shape_stride = index.get_shape_stride::<Dim4, Dim3>(&shape, &stride);

        assert_eq!(shape_stride.shape(), Dim3::new([4, 5, 6]));
        assert_eq!(shape_stride.stride(), Dim3::new([30, 6, 1]));
    }

    #[test]
    fn shape_stride_4d_index_1() {
        let shape = Dim4::new([3, 4, 5, 6]);
        let stride = Dim4::new([120, 30, 6, 1]);

        let index = Index1D::new(1);

        let shape_stride = index.get_shape_stride::<Dim4, Dim3>(&shape, &stride);

        assert_eq!(shape_stride.shape(), Dim3::new([3, 5, 6]));
        assert_eq!(shape_stride.stride(), Dim3::new([120, 6, 1]));
    }

    #[test]
    fn shape_stride_4d_index_2() {
        let shape = Dim4::new([3, 4, 5, 6]);
        let stride = Dim4::new([120, 30, 6, 1]);

        let index = Index2D::new(1);

        let shape_stride = index.get_shape_stride::<Dim4, Dim3>(&shape, &stride);

        assert_eq!(shape_stride.shape(), Dim3::new([3, 4, 6]));
        assert_eq!(shape_stride.stride(), Dim3::new([120, 30, 1]));
    }

    #[test]
    fn shape_stride_4d_index_3() {
        let shape = Dim4::new([3, 4, 5, 6]);
        let stride = Dim4::new([120, 30, 6, 1]);

        let index = Index3D::new(1);

        let shape_stride = index.get_shape_stride::<Dim4, Dim3>(&shape, &stride);

        assert_eq!(shape_stride.shape(), Dim3::new([3, 4, 5]));
        assert_eq!(shape_stride.stride(), Dim3::new([120, 30, 6]));
    }
}

use std::ops::Index;

use ruml_dim_impl::{Dim1, Dim2, Dim3, Dim4};
use ruml_index_impl::slice::{Slice1D, Slice2D, Slice3D, Slice4D};
use ruml_matrix_traits::{
    index::{IndexTrait, SliceTrait},
    matrix::MatrixSlice,
    memory::{Memory, OwnedMemory, ViewMemory},
    num::Num,
};

use crate::matrix::{
    CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix3D, CpuOwnedMatrix4D, CpuViewMatrix1D,
    CpuViewMatrix2D, CpuViewMatrix3D, CpuViewMatrix4D,
};
use crate::memory::CpuViewMemory;

macro_rules! impl_slice {
    ($owned_ty:ty, $view_ty:ty, $slice_ty:ty) => {
        impl<T: Num> MatrixSlice<$slice_ty> for $owned_ty {
            type Output<'a> = $view_ty where T: 'a;

            fn slice<'a>(&'a self, index: $slice_ty) -> Self::Output<'a> {
                let shape = self.shape();
                let stride = self.stride();

                let new_shape_stride = index.sliced_shape_stride(&shape, &stride);
                let offset = index.sliced_offset(&stride, 0);

                let data = self.data().to_view(offset);

                <$view_ty>::new(data, new_shape_stride.shape(), new_shape_stride.stride())
            }
        }

        impl<'a, T: Num> MatrixSlice<$slice_ty> for $view_ty {
            type Output<'b> = $view_ty where Self: 'b;

            fn slice(&self, index: $slice_ty) -> Self::Output<'_> {
                let shape = self.shape();
                let stride = self.stride();

                let new_shape_stride = index.sliced_shape_stride(&shape, &stride);
                let offset = index.sliced_offset(&stride, self.data().offset());

                let data = CpuViewMemory::new(self.data().reference(), offset);

                <$view_ty>::new(data, new_shape_stride.shape(), new_shape_stride.stride())
            }
        }
    };
}
impl_slice!(CpuOwnedMatrix1D<T>, CpuViewMatrix1D<'a, T>, Slice1D);
impl_slice!(CpuOwnedMatrix2D<T>, CpuViewMatrix2D<'a, T>, Slice2D);
impl_slice!(CpuOwnedMatrix3D<T>, CpuViewMatrix3D<'a, T>, Slice3D);
impl_slice!(CpuOwnedMatrix4D<T>, CpuViewMatrix4D<'a, T>, Slice4D);

macro_rules! impl_index {
    ($impl_ty:ty, $dim_ty:ty) => {
        impl<'a, T: Num> Index<$dim_ty> for $impl_ty {
            type Output = T;

            fn index(&self, index: $dim_ty) -> &Self::Output {
                let offset = dbg!(index.offset(&self.shape(), &self.stride()));
                &self.data().ptr_add(offset)
            }
        }
    };
}
impl_index!(CpuViewMatrix1D<'_, T>, Dim1);
impl_index!(CpuViewMatrix2D<'_, T>, Dim2);
impl_index!(CpuViewMatrix3D<'_, T>, Dim3);
impl_index!(CpuViewMatrix4D<'_, T>, Dim4);
impl_index!(CpuOwnedMatrix1D<T>, Dim1);
impl_index!(CpuOwnedMatrix2D<T>, Dim2);
impl_index!(CpuOwnedMatrix3D<T>, Dim3);
impl_index!(CpuOwnedMatrix4D<T>, Dim4);

#[cfg(test)]
mod matrix_index_test {
    use ruml_dim_impl::{Dim1, Dim2};
    use ruml_matrix_traits::matrix::OwnedMatrix;

    use crate::matrix::CpuOwnedMatrix1D;

    use super::CpuOwnedMatrix2D;

    #[test]
    fn test_index() {
        let owned = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], Dim1::new([4]));

        let view = owned.to_view();
        assert_eq!(owned[Dim1::new([0])], 1.);
        assert_eq!(view[Dim1::new([0])], 1.);

        assert_eq!(owned[Dim1::new([1])], 2.);
        assert_eq!(view[Dim1::new([1])], 2.);

        assert_eq!(owned[Dim1::new([2])], 3.);
        assert_eq!(view[Dim1::new([2])], 3.);

        assert_eq!(owned[Dim1::new([3])], 4.);
        assert_eq!(view[Dim1::new([3])], 4.);
    }

    #[test]
    fn test_index_2d() {
        let owned = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], Dim2::new([2, 2]));
        let view = owned.to_view();

        assert_eq!(owned[Dim2::new([0, 0])], 1.);
        assert_eq!(view[Dim2::new([0, 0])], 1.);

        assert_eq!(owned[Dim2::new([0, 1])], 2.);
        assert_eq!(view[Dim2::new([0, 1])], 2.);

        assert_eq!(owned[Dim2::new([1, 0])], 3.);
        assert_eq!(view[Dim2::new([1, 0])], 3.);

        assert_eq!(owned[Dim2::new([1, 1])], 4.);
        assert_eq!(view[Dim2::new([1, 1])], 4.);
    }
}

#[cfg(test)]
mod matrix_slice_test {
    use crate::matrix::{CpuOwnedMatrix2D, CpuOwnedMatrix3D, CpuOwnedMatrix4D};

    use ruml_dim_impl::{Dim1, Dim2, Dim3, Dim4};
    use ruml_index_impl::slice;
    use ruml_matrix_traits::matrix::{MatrixSlice, OwnedMatrix};

    use crate::matrix::CpuOwnedMatrix1D;

    #[test]
    fn slice_1d() {
        let owned = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4., 5., 6.], Dim1::new([6]));
        let sliced = owned.slice(slice!(..;2));

        assert_eq!(sliced.shape(), Dim1::new([3]));
        assert_eq!(sliced.stride(), Dim1::new([2]));

        assert_eq!(sliced[Dim1::new([0])], 1.);
        assert_eq!(sliced[Dim1::new([1])], 3.);
        assert_eq!(sliced[Dim1::new([2])], 5.);
    }

    #[test]
    fn slice_2d() {
        let v = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ];
        let owned = CpuOwnedMatrix2D::from_vec(v, Dim2::new([4, 4]));

        let sliced = owned.slice(slice!(1..3, ..3;2));
        let shape = sliced.shape();
        let stride = sliced.stride();

        assert_eq!(shape, Dim2::new([2, 2]));
        assert_eq!(stride, Dim2::new([4, 2]));

        assert_eq!(sliced[Dim2::new([0, 0])], 5.);
        assert_eq!(sliced[Dim2::new([0, 1])], 7.);
        assert_eq!(sliced[Dim2::new([1, 0])], 9.);
        assert_eq!(sliced[Dim2::new([1, 1])], 11.);
    }

    #[test]
    fn slice_3d() {
        let mut v = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    v.push((i * 16 + j * 4 + k) as f32);
                }
            }
        }

        let owned = CpuOwnedMatrix3D::from_vec(v, Dim3::new([4, 4, 4]));
        let slicesd = owned.slice(slice!(.., 2.., ..2));

        let shape = slicesd.shape();
        let stride = slicesd.stride();

        assert_eq!(shape, Dim3::new([4, 2, 2]));
        assert_eq!(stride, Dim3::new([16, 4, 1]));

        assert_eq!(slicesd[Dim3::new([0, 0, 0])], 8.);
        assert_eq!(slicesd[Dim3::new([0, 0, 1])], 9.);
        assert_eq!(slicesd[Dim3::new([0, 1, 0])], 12.);
        assert_eq!(slicesd[Dim3::new([0, 1, 1])], 13.);
        assert_eq!(slicesd[Dim3::new([1, 0, 0])], 24.);
        assert_eq!(slicesd[Dim3::new([1, 0, 1])], 25.);
        assert_eq!(slicesd[Dim3::new([1, 1, 0])], 28.);
        assert_eq!(slicesd[Dim3::new([1, 1, 1])], 29.);
        assert_eq!(slicesd[Dim3::new([2, 0, 0])], 40.);
        assert_eq!(slicesd[Dim3::new([2, 0, 1])], 41.);
        assert_eq!(slicesd[Dim3::new([2, 1, 0])], 44.);
        assert_eq!(slicesd[Dim3::new([2, 1, 1])], 45.);
        assert_eq!(slicesd[Dim3::new([3, 0, 0])], 56.);
        assert_eq!(slicesd[Dim3::new([3, 0, 1])], 57.);
        assert_eq!(slicesd[Dim3::new([3, 1, 0])], 60.);
        assert_eq!(slicesd[Dim3::new([3, 1, 1])], 61.);
    }

    #[test]
    fn slice_4d() {
        let mut v = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    for l in 0..4 {
                        v.push((i * 64 + j * 16 + k * 4 + l) as f32);
                    }
                }
            }
        }

        let owned = CpuOwnedMatrix4D::from_vec(v, Dim4::new([4, 4, 4, 4]));
        let slicesd = owned.slice(slice!(1..4;2, 1..;2, 1..2, 1..2));

        let shape = slicesd.shape();
        let stride = slicesd.stride();

        assert_eq!(shape, Dim4::new([2, 2, 1, 1]));
        assert_eq!(stride, Dim4::new([128, 32, 4, 1]));

        assert_eq!(slicesd[Dim4::new([0, 0, 0, 0])], 85.);
        assert_eq!(slicesd[Dim4::new([0, 1, 0, 0])], 117.);
        assert_eq!(slicesd[Dim4::new([1, 0, 0, 0])], 213.);
        assert_eq!(slicesd[Dim4::new([1, 1, 0, 0])], 245.);
    }
}

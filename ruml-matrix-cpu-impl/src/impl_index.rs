use std::ops::Index;

use ruml_dim_impl::{Dim1, Dim2, Dim3, Dim4};
use ruml_index_impl::slice::{Slice1D, Slice2D, Slice3D, Slice4D};
use ruml_matrix_traits::{
    index::{IndexTrait, SliceTrait},
    matrix::MatrixSlice,
    memory::{Memory, OwnedMemory},
    num::Num,
};

use crate::matrix::{
    CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix3D, CpuOwnedMatrix4D, CpuViewMatrix1D,
    CpuViewMatrix2D, CpuViewMatrix3D, CpuViewMatrix4D,
};

macro_rules! impl_slice {
    ($owned_ty:ty, $view_ty:ty, $slice_ty:ty) => {
        impl<T: Num> MatrixSlice<$slice_ty> for $owned_ty {
            type Output = $view_ty;

            fn slice(&self, index: $slice_ty) -> Self::Output {
                let shape = self.shape();
                let stride = self.stride();

                let new_shape_stride = index.sliced_shape_stride(&shape, &stride);

                let data = self.data().to_view(0);

                <$view_ty>::new(data, new_shape_stride.shape(), new_shape_stride.stride())
            }
        }

        impl<T: Num> MatrixSlice<$slice_ty> for $view_ty {
            type Output = $view_ty;

            fn slice(&self, index: $slice_ty) -> Self::Output {
                let shape = self.shape();
                let stride = self.stride();

                let new_shape_stride = index.sliced_shape_stride(&shape, &stride);

                let data = self.data().clone();

                <$view_ty>::new(data, new_shape_stride.shape(), new_shape_stride.stride())
            }
        }
    };
}
impl_slice!(CpuOwnedMatrix1D<T>, CpuViewMatrix1D<T>, Slice1D);
impl_slice!(CpuOwnedMatrix2D<T>, CpuViewMatrix2D<T>, Slice2D);
impl_slice!(CpuOwnedMatrix3D<T>, CpuViewMatrix3D<T>, Slice3D);
impl_slice!(CpuOwnedMatrix4D<T>, CpuViewMatrix4D<T>, Slice4D);

macro_rules! impl_index {
    ($impl_ty:ty, $dim_ty:ty) => {
        impl<T: Num> Index<$dim_ty> for $impl_ty {
            type Output = T;

            fn index(&self, index: $dim_ty) -> &Self::Output {
                let offset = dbg!(index.offset(&self.shape(), &self.stride()));
                &self.data().ptr_add(offset)
            }
        }
    };
}
impl_index!(CpuViewMatrix1D<T>, Dim1);
impl_index!(CpuViewMatrix2D<T>, Dim2);
impl_index!(CpuViewMatrix3D<T>, Dim3);
impl_index!(CpuViewMatrix4D<T>, Dim4);
impl_index!(CpuOwnedMatrix1D<T>, Dim1);
impl_index!(CpuOwnedMatrix2D<T>, Dim2);
impl_index!(CpuOwnedMatrix3D<T>, Dim3);
impl_index!(CpuOwnedMatrix4D<T>, Dim4);

#[cfg(test)]
mod matrix_index_test {
    use ruml_dim_impl::{Dim1, Dim2};
    use ruml_matrix_traits::matrix::{Matrix, OwnedMatrix};

    use crate::matrix::CpuOwnedMatrix1D;

    use super::CpuOwnedMatrix2D;

    #[test]
    fn test_index() {
        println!("test_index");
        let owned = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], Dim1::new([4]));
        let stride = owned.stride();
        println!("stride: {:?}", stride);
        let view = owned.to_view();
        assert_eq!(owned[Dim1::new([0])], 1.);
        // assert_eq!(view[Dim1::new([0])], 1.);

        assert_eq!(owned[Dim1::new([1])], 2.);
        // assert_eq!(view[Dim1::new([1])], 2.);

        assert_eq!(owned[Dim1::new([2])], 3.);
        // assert_eq!(view[Dim1::new([2])], 3.);

        assert_eq!(owned[Dim1::new([3])], 4.);
        // assert_eq!(view[Dim1::new([3])], 4.);
    }

    #[test]
    fn test_index_2d() {
        let owned = CpuOwnedMatrix2D::from_vec(vec![1., 2., 3., 4.], Dim2::new([2, 2]));
        //     let view = owned.to_view();

        assert_eq!(owned[Dim2::new([0, 0])], 1.);
        //     assert_eq!(view[Dim2::new([0, 0])], 1.);

        assert_eq!(owned[Dim2::new([0, 1])], 2.);
        //     assert_eq!(view[Dim2::new([0, 1])], 2.);

        assert_eq!(owned[Dim2::new([1, 0])], 3.);
        //     assert_eq!(view[Dim2::new([1, 0])], 3.);

        assert_eq!(owned[Dim2::new([1, 1])], 4.);
        //     assert_eq!(view[Dim2::new([1, 1])], 4.);
    }
}

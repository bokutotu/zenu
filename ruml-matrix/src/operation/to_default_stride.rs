use crate::{
    cpu_memory::{CpuOwnedMemory, CpuViewMemory, CpuViewMutMemory},
    dim,
    dim_impl::{Dim1, Dim2, Dim3, Dim4},
    index_impl::Index0D,
    matrix::{
        IndexAxis, IndexAxisMut, IndexItem, IndexItemAsign, MatrixBase, OwnedMatrix, ToViewMatrix,
        ToViewMutMatrix,
    },
    matrix_impl::Matrix,
    num::Num,
};

use super::{copy_from::CopyFrom, zeros::Zeros};

pub trait ToDefaultStride: MatrixBase {
    type Output: OwnedMatrix;
    fn to_default_stride(&self) -> Self::Output;
}

impl<T: Num> ToDefaultStride for Matrix<CpuOwnedMemory<T>, Dim1> {
    type Output = Matrix<CpuOwnedMemory<T>, Dim1>;
    fn to_default_stride(&self) -> Self::Output {
        let mut output = Self::Output::zeros(self.shape_stride().shape());
        for idx in 0..self.shape_stride().shape()[0] {
            output
                .to_view_mut()
                .index_item_asign(dim!(idx), self.index_item(dim!(idx)));
        }
        output
    }
}

impl<T: Num> ToDefaultStride for Matrix<CpuViewMemory<'_, T>, Dim1> {
    type Output = Matrix<CpuOwnedMemory<T>, Dim1>;
    fn to_default_stride(&self) -> Self::Output {
        let mut output = Self::Output::zeros(self.shape_stride().shape());
        for idx in 0..self.shape_stride().shape()[0] {
            output
                .to_view_mut()
                .index_item_asign(dim!(idx), self.index_item(dim!(idx)));
        }
        output
    }
}

macro_rules! impl_to_default_stride {
    ($dim:ty, $($memory:tt)+) => {
        impl<T: Num> ToDefaultStride for Matrix<$($memory)+, $dim> {
            type Output = Matrix<CpuOwnedMemory<T>, $dim>;
            fn to_default_stride(&self) -> Self::Output {
                let mut output = Self::Output::zeros(self.shape_stride().shape());
                for idx in 0..self.shape_stride().shape()[0] {
                    let default_stride = self.index_axis(Index0D(idx)).to_default_stride();
                    output.index_axis_mut(Index0D(idx)).copy_from(&default_stride.to_view());
                }
                output
            }
        }
    };
}
impl_to_default_stride!(Dim2, CpuOwnedMemory<T>);
impl_to_default_stride!(Dim2, CpuViewMemory<'_, T>);
impl_to_default_stride!(Dim2, CpuViewMutMemory<'_, T>);
impl_to_default_stride!(Dim3, CpuOwnedMemory<T>);
impl_to_default_stride!(Dim3, CpuViewMemory<'_, T>);
impl_to_default_stride!(Dim3, CpuViewMutMemory<'_, T>);
impl_to_default_stride!(Dim4, CpuOwnedMemory<T>);
impl_to_default_stride!(Dim4, CpuViewMemory<'_, T>);
impl_to_default_stride!(Dim4, CpuViewMutMemory<'_, T>);

#[cfg(test)]
mod to_default_stride {
    use crate::{
        dim::default_stride,
        matrix::{MatrixSlice, OwnedMatrix},
        matrix_impl::{CpuOwnedMatrix1D, CpuOwnedMatrix2D},
        slice,
    };

    use super::*;

    #[test]
    fn test_1d() {
        // 0 t0 16 f32 vec
        let v = vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        ];

        let m = CpuOwnedMatrix1D::from_vec(v.clone(), dim!(16));
        let sliced = m.slice(slice!(..;2));
        let default_strided = sliced.to_default_stride();

        assert_eq!(
            default_strided.shape_stride().stride(),
            default_stride(default_strided.shape_stride().shape())
        );

        assert_eq!(default_strided.index_item(dim!(0)), 0.);
        assert_eq!(default_strided.index_item(dim!(1)), 2.);
        assert_eq!(default_strided.index_item(dim!(2)), 4.);
        assert_eq!(default_strided.index_item(dim!(3)), 6.);
        assert_eq!(default_strided.index_item(dim!(4)), 8.);
    }

    #[test]
    fn test_2d() {
        // 0 t0 16 f32 vec
        let v = vec![
            0., 1., 2., 3., 4., 5., 6., 7., //
            8., 9., 10., 11., 12., 13., 14., 15.,
        ];

        let m = CpuOwnedMatrix2D::from_vec(v.clone(), dim!(4, 4));
        let sliced = m.slice(slice!(..;2, ..;2));
        let default_strided = sliced.to_default_stride();

        assert_eq!(
            default_strided.shape_stride().stride(),
            default_stride(default_strided.shape_stride().shape())
        );

        assert_eq!(default_strided.index_item(dim!(0, 0)), 0.);
        assert_eq!(default_strided.index_item(dim!(0, 1)), 2.);
        assert_eq!(default_strided.index_item(dim!(1, 0)), 8.);
        assert_eq!(default_strided.index_item(dim!(1, 1)), 10.);
    }
}

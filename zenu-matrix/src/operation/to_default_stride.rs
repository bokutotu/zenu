use crate::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory_impl::{OwnedMem, ViewMem},
    num::Num,
};

use super::{copy_from::CopyFrom, zeros::Zeros};

pub trait ToDefaultStride<T: Num> {
    fn to_default_stride<SD: DimTrait>(source: Matrix<ViewMem<T>, SD>) -> Self;
}

impl<T> ToDefaultStride<T> for Matrix<OwnedMem<T>, DimDyn>
where
    T: Num,
{
    fn to_default_stride<SD>(source: Matrix<ViewMem<T>, SD>) -> Self
    where
        SD: DimTrait,
    {
        let mut output = <Self as Zeros>::zeros(source.shape().slice());
        {
            let mut output_view_mut = output.to_view_mut();
            let source_dyn = source.into_dyn_dim();
            output_view_mut.copy_from(&source_dyn);
        }
        output
    }
}
#[cfg(test)]
mod to_default_stride {
    use crate::{
        dim::default_stride,
        matrix::{IndexItem, MatrixSlice, OwnedMatrix},
        matrix_impl::{OwnedMatrix1D, OwnedMatrix2D, OwnedMatrixDyn},
        slice,
    };

    use super::*;

    #[test]
    fn test_1d() {
        // 0 t0 16 f32 vec
        let v = vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        ];

        let m = OwnedMatrix1D::from_vec(v.clone(), [16]);
        let sliced = m.slice(slice!(..;2));
        let default_strided: OwnedMatrixDyn<f32> = ToDefaultStride::to_default_stride(sliced);

        assert_eq!(
            default_strided.shape_stride().stride(),
            default_stride(default_strided.shape_stride().shape())
        );

        assert_eq!(default_strided.index_item([0]), 0.);
        assert_eq!(default_strided.index_item([1]), 2.);
        assert_eq!(default_strided.index_item([2]), 4.);
        assert_eq!(default_strided.index_item([3]), 6.);
        assert_eq!(default_strided.index_item([4]), 8.);
    }

    #[test]
    fn test_2d() {
        // 0 t0 16 f32 vec
        let v = vec![
            0., 1., 2., 3., 4., 5., 6., 7., //
            8., 9., 10., 11., 12., 13., 14., 15.,
        ];

        let m = OwnedMatrix2D::from_vec(v.clone(), [4, 4]);
        let sliced = m.slice(slice!(..;2, ..;2));
        let default_strided: OwnedMatrixDyn<f32> = ToDefaultStride::to_default_stride(sliced);

        assert_eq!(
            default_strided.shape_stride().stride(),
            default_stride(default_strided.shape_stride().shape())
        );

        assert_eq!(default_strided.index_item([0, 0]), 0.);
        assert_eq!(default_strided.index_item([0, 1]), 2.);
        assert_eq!(default_strided.index_item([1, 0]), 8.);
        assert_eq!(default_strided.index_item([1, 1]), 10.);
    }
}

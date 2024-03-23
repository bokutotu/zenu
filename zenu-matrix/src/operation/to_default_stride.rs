use crate::{
    constructor::zeros::Zeros,
    dim::{DimDyn, DimTrait},
    matrix::{ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::ToViewMemory,
    memory_impl::OwnedMem,
    num::Num,
};

use super::copy_from::CopyFrom;

pub trait ToDefaultStride<T: Num> {
    fn to_default_stride(&self) -> Matrix<OwnedMem<T>, DimDyn>;
}

impl<T, M, D: DimTrait> ToDefaultStride<T> for Matrix<M, D>
where
    T: Num,
    M: ToViewMemory<Item = T>,
{
    fn to_default_stride(&self) -> Matrix<OwnedMem<T>, DimDyn> {
        let mut output: Matrix<OwnedMem<T>, DimDyn> = Zeros::zeros_like(self.to_view());
        {
            let mut output_view_mut = output.to_view_mut();
            output_view_mut.copy_from(&self.to_view().into_dyn_dim());
        }
        output
    }
}
#[cfg(test)]
mod to_default_stride {
    use crate::{
        dim::default_stride,
        matrix::{IndexItem, MatrixBase, MatrixSlice, OwnedMatrix},
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
        // let default_strided: OwnedMatrixDyn<f32> = ToDefaultStride::to_default_stride(&sliced);
        let default_strided = sliced.to_default_stride();

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
        let default_strided: OwnedMatrixDyn<f32> = ToDefaultStride::to_default_stride(&sliced);

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

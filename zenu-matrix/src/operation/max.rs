use crate::{
    blas::Blas,
    dim::{DimDyn, DimTrait},
    matrix::{AsPtr, BlasMatrix, IndexItem, MatrixBase},
    matrix_impl::Matrix,
    memory_impl::{OwnedMem, ViewMem},
    num::Num,
};

use super::to_default_stride::ToDefaultStride;

pub trait MaxIdx<T, D> {
    fn max_idx(self) -> DimDyn;
    fn max(self) -> T;
}

impl<'a, T, D> MaxIdx<T, D> for Matrix<ViewMem<'a, T>, D>
where
    T: Num,
    D: DimTrait,
{
    fn max_idx(self) -> DimDyn {
        let default_stride: Matrix<OwnedMem<T>, DimDyn> = ToDefaultStride::to_default_stride(self);
        let idx = <Self as BlasMatrix>::Blas::amax(
            default_stride.shape().num_elm(),
            default_stride.as_ptr(),
            default_stride.stride()[default_stride.shape().len() - 1],
        );
        default_stride.shape_stride().get_dim_by_offset(idx)
    }

    fn max(self) -> T {
        let s = self.into_dyn_dim();
        let idx = s.clone().max_idx();
        s.index_item(idx)
    }
}

#[cfg(test)]
mod max_idx {
    use crate::{
        matrix::{MatrixSlice, OwnedMatrix, ToViewMatrix},
        matrix_impl::{OwnedMatrix1D, OwnedMatrix2D, OwnedMatrix3D},
        operation::max::MaxIdx,
        slice,
    };

    #[test]
    fn default_1d() {
        let a = OwnedMatrix1D::from_vec(vec![0., 1., 2., 3.], [4]);
        assert_eq!(a.to_view().max_idx(), [3].into());
    }

    #[test]
    fn default_2d() {
        let a = OwnedMatrix2D::from_vec(vec![0., 1., 2., 3.], [2, 2]);
        assert_eq!(a.to_view().max_idx(), [1, 1].into());
    }

    #[test]
    fn sliced_3d() {
        let mut v = Vec::new();
        for i in 0..8 * 8 * 8 {
            v.push(i as f32);
        }
        let a = OwnedMatrix3D::from_vec(v, [8, 8, 8]);
        let sliced = a.slice(slice!(..;3, ..;4, ..;2));
        assert_eq!(sliced.max_idx(), [2, 1, 3].into());
    }
}

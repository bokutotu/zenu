use crate::{
    blas::Blas,
    dim::{DimDyn, DimTrait},
    index::index_dyn_impl::Index,
    matrix::{
        AsMutPtr, AsPtr, IndexAxisDyn, IndexAxisMutDyn, MatrixBase, ViewMatrix, ViewMutMatix,
    },
    matrix_impl::Matrix,
    memory::{Memory, ViewMemory, ViewMutMemory},
    num::Num,
};

pub trait CopyFrom<RHS>: ViewMutMatix
where
    RHS: ViewMatrix,
{
    fn copy_from(&mut self, rhs: &RHS);
}

impl<T, V, VM> CopyFrom<Matrix<V, DimDyn>> for Matrix<VM, DimDyn>
where
    T: Num,
    VM: ViewMutMemory<Item = T>,
    V: ViewMemory<Item = T>,
{
    fn copy_from(&mut self, rhs: &Matrix<V, DimDyn>) {
        copy(self, rhs);
    }
}

fn copy<T, VM, V>(to: &mut Matrix<VM, DimDyn>, source: &Matrix<V, DimDyn>)
where
    T: Num,
    VM: ViewMutMemory<Item = T>,
    V: ViewMemory<Item = T>,
{
    assert_eq!(to.shape(), source.shape());

    if to.shape_stride().is_contiguous() && source.shape_stride().is_contiguous()
        || to.shape().len() == 1
    {
        <VM as Memory>::Blas::copy(
            to.shape().num_elm(),
            source.as_ptr(),
            source.stride()[to.shape().len() - 1],
            to.as_mut_ptr() as *mut _,
            to.stride()[to.shape().len() - 1],
        );
        return;
    }

    match to.shape().len() {
        2 => {
            for i in 0..to.shape()[0] {
                let mut to_ = to.index_axis_mut_dyn(Index::new(0, i));
                let source_ = source.index_axis_dyn(Index::new(0, i));
                copy(&mut to_, &source_);
            }
        }
        3 => {
            for i in 0..to.shape()[0] {
                let mut to_ = to.index_axis_mut_dyn(Index::new(0, i));
                let source_ = source.index_axis_dyn(Index::new(0, i));
                copy(&mut to_, &source_);
            }
        }
        4 => {
            for i in 0..to.shape()[0] {
                let mut to_ = to.index_axis_mut_dyn(Index::new(0, i));
                let source_ = source.index_axis_dyn(Index::new(0, i));
                copy(&mut to_, &source_);
            }
        }
        _ => panic!("Not implemented"),
    }
}

#[cfg(test)]
mod deep_copy {
    use super::*;
    use crate::{
        matrix::{
            IndexItem, MatrixSlice, MatrixSliceMut, OwnedMatrix, ToViewMatrix, ToViewMutMatrix,
        },
        matrix_impl::{CpuOwnedMatrix1D, CpuOwnedMatrix2D},
        slice,
    };

    #[test]
    fn default_stride_1d() {
        let a = vec![0f32; 6];
        let b = vec![1f32, 2., 3., 4., 5., 6.];

        let mut a = CpuOwnedMatrix1D::from_vec(a, [6]);
        let b = CpuOwnedMatrix1D::from_vec(b, [6]);

        let a_view_mut = a.to_view_mut();

        a_view_mut
            .into_dyn_dim()
            .to_view_mut()
            .copy_from(&b.to_view().into_dyn_dim());

        assert_eq!(a.index_item([0]), 1.);
        assert_eq!(a.index_item([1]), 2.);
        assert_eq!(a.index_item([2]), 3.);
        assert_eq!(a.index_item([3]), 4.);
        assert_eq!(a.index_item([4]), 5.);
        assert_eq!(a.index_item([5]), 6.);
    }

    #[test]
    fn sliced_1d() {
        let a = vec![0f32; 6];
        let v = vec![0f32, 1., 2., 3., 4., 5.];

        let mut a = CpuOwnedMatrix1D::from_vec(a.clone(), [6]);
        let v = CpuOwnedMatrix1D::from_vec(v, [6]);

        let a_sliced = a.slice_mut(slice!(..;2));
        let v_sliced = v.slice(slice!(0..3));

        a_sliced.into_dyn_dim().copy_from(&v_sliced.into_dyn_dim());
        assert_eq!(a.index_item([0]), 0.);
        assert_eq!(a.index_item([1]), 0.);
        assert_eq!(a.index_item([2]), 1.);
        assert_eq!(a.index_item([3]), 0.);
        assert_eq!(a.index_item([4]), 2.);
        assert_eq!(a.index_item([5]), 0.);
    }

    #[test]
    fn defualt_stride_2d() {
        let a = vec![0f32; 6];
        let b = vec![1f32, 2., 3., 4., 5., 6.];

        let mut a = CpuOwnedMatrix2D::from_vec(a, [2, 3]);
        let b = CpuOwnedMatrix2D::from_vec(b, [2, 3]);

        let a_view_mut = a.to_view_mut();

        a_view_mut
            .into_dyn_dim()
            .to_view_mut()
            .copy_from(&b.to_view().into_dyn_dim());

        assert_eq!(a.index_item([0, 0]), 1.);
        assert_eq!(a.index_item([0, 1]), 2.);
        assert_eq!(a.index_item([0, 2]), 3.);
        assert_eq!(a.index_item([1, 0]), 4.);
        assert_eq!(a.index_item([1, 1]), 5.);
        assert_eq!(a.index_item([1, 2]), 6.);
    }

    #[test]
    fn sliced_2d() {
        let a = vec![0f32; 12];
        let v = vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.];
        let mut a = CpuOwnedMatrix2D::from_vec(a.clone(), [3, 4]);
        let v = CpuOwnedMatrix2D::from_vec(v, [3, 4]);

        let a_sliced = a.slice_mut(slice!(0..2, 0..3));
        let v_sliced = v.slice(slice!(1..3, 1..4));

        a_sliced.into_dyn_dim().copy_from(&v_sliced.into_dyn_dim());
        assert_eq!(a.index_item([0, 0]), 5.);
        assert_eq!(a.index_item([0, 1]), 6.);
        assert_eq!(a.index_item([0, 2]), 7.);
        assert_eq!(a.index_item([0, 3]), 0.);
        assert_eq!(a.index_item([1, 0]), 9.);
        assert_eq!(a.index_item([1, 1]), 10.);
        assert_eq!(a.index_item([1, 2]), 11.);
        assert_eq!(a.index_item([2, 3]), 0.);
    }
}

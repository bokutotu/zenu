use crate::{
    blas::Blas,
    dim::{DimDyn, DimTrait},
    index::index_dyn_impl::Index,
    matrix::{
        AsMutPtr, AsPtr, IndexAxisDyn, IndexAxisMutDyn, MatrixBase, ToViewMatrix, ToViewMutMatrix,
    },
    matrix_impl::Matrix,
    memory::{Memory, ToViewMemory, ToViewMutMemory},
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
    shape_stride::ShapeStride,
};

pub trait CopyFrom<RHS>: ToViewMutMatrix
where
    RHS: ToViewMatrix,
{
    fn copy_from(&mut self, rhs: &RHS);
}

impl<T, V, VM> CopyFrom<Matrix<V, DimDyn>> for Matrix<VM, DimDyn>
where
    T: Num,
    VM: ToViewMutMemory<Item = T>,
    V: ToViewMemory<Item = T>,
{
    fn copy_from(&mut self, rhs: &Matrix<V, DimDyn>) {
        assert_eq!(self.shape().slice(), rhs.shape().slice(), "Shape mismatch");
        copy(self.to_view_mut(), rhs.to_view());
    }
}

fn check_can_use_blas<D: DimTrait>(to: ShapeStride<D>, source: ShapeStride<D>) -> bool {
    if to.shape().len() == 1 {
        return true;
    }
    if !to.is_default_stride() || !source.is_default_stride() {
        return false;
    }
    if to.is_transposed() != source.is_transposed() {
        return false;
    }
    if to.is_contiguous() && source.is_contiguous() {
        return true;
    }
    false
}

fn copy<T: Num>(mut to: Matrix<ViewMutMem<T>, DimDyn>, source: Matrix<ViewMem<T>, DimDyn>) {
    if to.shape().is_empty() {
        unsafe {
            to.as_mut_ptr().write(source.as_ptr().read());
        }
        return;
    }
    if check_can_use_blas(to.shape_stride(), source.shape_stride()) {
        let s_stride_max = source.shape_stride().min_stride();
        let t_stride_max = to.shape_stride().min_stride();
        <ViewMutMem<T> as Memory>::Blas::copy(
            to.shape().num_elm(),
            source.as_ptr(),
            s_stride_max,
            to.as_mut_ptr() as *mut _,
            t_stride_max,
        );
    } else {
        for idx in 0..to.shape()[0] {
            let to_ = to.index_axis_mut_dyn(Index::new(0, idx));
            let source_ = source.index_axis_dyn(Index::new(0, idx));
            copy(to_, source_);
        }
    }
}

#[cfg(test)]
mod deep_copy {
    use super::*;
    use crate::{
        matrix::{
            IndexItem, MatrixSlice, MatrixSliceMut, OwnedMatrix, ToViewMatrix, ToViewMutMatrix,
        },
        matrix_impl::{OwnedMatrix1D, OwnedMatrix2D},
        slice,
    };

    #[test]
    fn default_stride_1d() {
        let a = vec![0f32; 6];
        let b = vec![1f32, 2., 3., 4., 5., 6.];

        let mut a = OwnedMatrix1D::from_vec(a, [6]);
        let b = OwnedMatrix1D::from_vec(b, [6]);

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

        let mut a = OwnedMatrix1D::from_vec(a.clone(), [6]);
        let v = OwnedMatrix1D::from_vec(v, [6]);

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

        let mut a = OwnedMatrix2D::from_vec(a, [2, 3]);
        let b = OwnedMatrix2D::from_vec(b, [2, 3]);

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
        let mut a = OwnedMatrix2D::from_vec(a.clone(), [3, 4]);
        let v = OwnedMatrix2D::from_vec(v, [3, 4]);

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

use crate::{
    blas::Blas,
    dim::{DimTrait, LessDimTrait},
    dim_impl::{Dim1, Dim2},
    index::IndexAxisTrait,
    index_impl::{Index1D, Index2D, Index3D},
    matrix::{IndexAxis, IndexAxisMut, MatrixBase, ViewMatrix, ViewMutMatix},
    memory::{Memory, ViewMemory, ViewMutMemory},
    num::Num,
};

pub trait CopyFrom<'a, RHS: ViewMatrix, D: DimTrait, I: IndexAxisTrait, N: Num>:
    ViewMutMatix
{
    fn copy_from(&mut self, rhs: RHS);
}

impl<'a, V, VM, N> CopyFrom<'a, V, Dim1, Index1D, N> for VM
where
    N: Num,
    V: ViewMatrix + MatrixBase<Dim = Dim1>,
    VM: ViewMutMatix + MatrixBase<Dim = Dim1>,
    VM::Memory: ViewMutMemory + Memory<Item = N>,
    V::Memory: Memory<Item = N> + ViewMemory,
{
    fn copy_from(&mut self, rhs: V) {
        let self_shape_stride = self.shape_stride();
        let rhs_shape_stride = rhs.shape_stride();

        if self_shape_stride.shape() != rhs_shape_stride.shape() {
            panic!("shape is not equal");
        }

        <<VM as MatrixBase>::Memory as Memory>::Blas::copy(
            rhs_shape_stride.shape().len(),
            rhs.memory().as_ptr(),
            self_shape_stride.stride().len(),
            self.memory().as_mut_ptr(),
            self_shape_stride.shape().len(),
        )
    }
}

impl<'a, V, VM, N> CopyFrom<'a, V, Dim2, Index1D, N> for VM
where
    N: Num,
    V: ViewMatrix + MatrixBase<Dim = Dim2> + IndexAxis<Index1D> + 'a,
    VM: ViewMutMatix + MatrixBase<Dim = Dim2> + IndexAxisMut<Index1D> + 'a,
    VM::Memory: ViewMutMemory + Memory<Item = N>,
    V::Memory: Memory<Item = N> + ViewMemory,
    <VM as IndexAxisMut<Index1D>>::Output<'a>:
        CopyFrom<'a, <V as IndexAxis<Index1D>>::Output<'a>, Dim1, Index1D, N>,
{
    fn copy_from<'b: 'a>(&'b mut self, rhs: V) {
        let self_shape_stride = self.shape_stride();
        let rhs_shape_stride = rhs.shape_stride();

        if self_shape_stride.shape() != rhs_shape_stride.shape() {
            panic!("shape is not equal");
        }

        for i in 0..self_shape_stride.shape().len() {
            let mut s_sliced = self.index_axis_mut(Index1D::new(i));
            let r_sliced = rhs.index_axis(Index1D::new(i));

            s_sliced.copy_from(r_sliced);
        }
    }
}

// #[cfg(test)]
// mod deep_copy {
//     use super::*;
//     use crate::{
//         dim,
//         matrix::{IndexItem, MatrixSlice, MatrixSliceMut, OwnedMatrix},
//         matrix_impl::CpuOwnedMatrix2D,
//         slice,
//     };
//
//     #[test]
//     fn defualt_stride() {
//         let a = vec![0f32; 6];
//         let b = vec![1f32, 2., 3., 4., 5., 6.];
//
//         let mut a = CpuOwnedMatrix2D::from_vec(a, dim!(2, 3));
//         let b = CpuOwnedMatrix2D::from_vec(b, dim!(2, 3));
//
//         a.to_view_mut().copy_from(&b.to_view());
//
//         assert_eq!(a.index_item(dim!(0, 0)), 1.);
//         assert_eq!(a.index_item(dim!(0, 1)), 2.);
//         assert_eq!(a.index_item(dim!(0, 2)), 3.);
//         assert_eq!(a.index_item(dim!(1, 0)), 4.);
//         assert_eq!(a.index_item(dim!(1, 1)), 5.);
//         assert_eq!(a.index_item(dim!(1, 2)), 6.);
//     }
//
//     #[test]
//     fn sliced() {
//         let a = vec![0f32; 6];
//         let v = vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
//         let mut a = CpuOwnedMatrix2D::from_vec(a.clone(), dim!(3, 4));
//         let v = CpuOwnedMatrix2D::from_vec(v, dim!(3, 4));
//
//         let a_sliced = a.slice_mut(slice!(0..2, 0..3));
//         let v_sliced = v.slice(slice!(1..3, 1..4));
//
//         a_sliced.copy_from(&v_sliced);
//         assert_eq!(a.index_item(dim!(0, 0)), 0.);
//         assert_eq!(a.index_item(dim!(0, 1)), 1.);
//         assert_eq!(a.index_item(dim!(0, 2)), 2.);
//         assert_eq!(a.index_item(dim!(0, 3)), 0.);
//         assert_eq!(a.index_item(dim!(1, 0)), 4.);
//         assert_eq!(a.index_item(dim!(1, 1)), 5.);
//         assert_eq!(a.index_item(dim!(1, 2)), 6.);
//         assert_eq!(a.index_item(dim!(2, 3)), 0.);
//     }
// }

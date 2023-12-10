// use std::ops::{Index, IndexMut};
//
// use crate::{
//     blas::Blas,
//     cpu_blas::CpuBlas,
//     dim::{default_stride, DimTrait},
//     index::SliceTrait,
//     index_impl::Index3D,
//     matrix::{MatrixBase, MatrixSlice, MatrixSliceMut, ViewMatrix, ViewMutMatix},
//     matrix_impl::Matrix,
//     memory::{Memory, ToViewMutMemory, ViewMemory, ViewMutMemory},
//     num::Num,
// };

// pub trait DeepCopy<T: Num, M: ViewMatrix<Dim = D>, D: DimTrait>: ViewMutMatix<Dim = D>
// pub trait DeepCopy<T, M, D, S>: ViewMutMatix<Dim = D> + MatrixSliceMut<D, S>
// where
//     T: Num,
//     S: SliceTrait<Dim = D>,
//     M: ViewMatrix<Dim = D> + MatrixSlice<S, D>,
//     D: DimTrait,
//     Self::Memory: ViewMutMemory + ToViewMutMemory,
//     Self::Memory: Memory<Item = T>,
//     <M as MatrixBase>::Memory: ViewMemory,
//     <M as MatrixBase>::Memory: Memory<Item = T>,
//     CpuBlas<T>: Blas<T>,
// {
//     type Blas: Blas<<<Self as MatrixBase>::Memory as Memory>::Item>;
//     fn copy_from(&self, m: &M) {
//         let self_shape_stride = self.shape_stride();
//         let m_shape_stride = m.shape_stride();
//
//         let self_shape = self_shape_stride.shape();
//         let self_stride = self_shape_stride.stride();
//
//         let m_shape = m_shape_stride.shape();
//         let m_stride = m_shape_stride.stride();
//
//         if self_shape.len() == 1 {
//             let ptr_mut = self.view_mut_memory().as_mut_ptr();
//             let ptr = m.memory().as_ptr();
//
//             let inc_mut = self_stride[0];
//             let inc = m_stride[0];
//
//             Self::Blas::copy(self_shape[0], ptr, inc, ptr_mut, inc_mut);
//         }
//
//         let num_axis = self_shape.len();
//
//         for i in 0..self_shape[0] {
//             if num_axis == 4 {
//                 let index_axis = Index3D::new(i);
//                 self.index_mut(index_axis).copy_from(m.index(index_axis));
//             }
//         }
//     }
// }

// impl<T, M, D, Me, S> DeepCopy<T, M, D, S> for Matrix<Me, D> + MatrixSliceMut<S, D>
// impl<T, M, D, Me, S> DeepCopy<T, M, D, S> for Matrix<Me, D>
// where
//     Me: ViewMutMemory,
//     T: Num,
//     M: ViewMatrix<Dim = D>,
//     D: DimTrait,
//     S: SliceTrait<Dim = D>,
//     Self::Memory: ViewMutMemory,
//     Self::Memory: Memory<Item = T>,
//     <M as MatrixBase>::Memory: ViewMemory,
//     <M as MatrixBase>::Memory: Memory<Item = T>,
//     CpuBlas<T>: Blas<T>,
// {
//     type Blas = CpuBlas<T>;
// }

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

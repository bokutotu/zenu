// use crate::{
//     blas::Blas,
//     dim::DimTrait,
//     matrix::{MatrixBase, ViewMatrix, ViewMutMatix},
//     memory::{Memory, ViewMemory, ViewMutMemory},
//     num::Num,
// };
//
// pub trait MatrixDeepCopy<T, RHS, D>: ViewMutMatix<Dim = D>
// where
//     D: DimTrait,
//     T: Num,
//     Self::Memory: ViewMutMemory<Item = T>,
//     RHS: ViewMatrix + MatrixBase<Dim = D>,
//     <RHS as MatrixBase>::Memory: ViewMemory<Item = T>,
// {
//     type Blas: Blas<T>;
//
//     fn deep_copy(&self, rhs: RHS) {
//         if self.shape_stride().shape() != rhs.shape_stride().shape() {
//             panic!("self.shape_stride().shape() != rhs.shape_stride().shape()");
//         }
//         if !(self.is_default_stride() && rhs.is_default_stride()) {
//             panic!("!(self.default_stride() && rhs.default_stride())");
//         }
//
//         // call copy
//         Self::Blas::copy(
//             self.shape_stride().shape().num_elm(),
//             rhs.memory().as_ptr(),
//             1,
//             self.view_mut_memory().as_mut_ptr(),
//             1,
//         );
//     }
// }
//
// pub trait MatrixMul<T, Dim, RHS, LHS>: ViewMutMatix + MatrixDeepCopy<T, RHS, Dim>
// where
//     T: Num,
//     Dim: DimTrait,
//     RHS: ViewMatrix<Dim = Dim>,
//     LHS: ViewMatrix,
//     Self::Memory: ViewMutMemory<Item = T>,
//     Self::Dim: DimTrait,
//     <RHS as MatrixBase>::Memory: ViewMemory<Item = T>,
//     <LHS as MatrixBase>::Memory: ViewMemory<Item = T>,
// {
//     type Blas: Blas<T>;
//
//     fn mul(&self, rhs: RHS, lhs: LHS) {
//         if !(self.is_default_stride() && rhs.is_default_stride() && lhs.is_default_stride()) {
//             unimplemented!()
//         }
//
//         // TODO: この部分はコンパイル時にチェックするようにしたい
//         if rhs.shape_stride().shape().len() < lhs.shape_stride().shape().len() {
//             panic!("rhs.shape_stride().shape().len() < lhs.shape_stride().shape().len()");
//         }
//
//         // copy rhs data to self
//         self.deep_copy(rhs);
//         // scalar
//         // call scal
//         if lhs.shape_stride().shape().len() == 0 {
//             <Self as MatrixMul<T, Dim, RHS, LHS>>::Blas::scal(
//                 self.shape_stride().shape().num_elm(),
//                 unsafe { *lhs.memory().as_ptr() },
//                 self.view_mut_memory().as_mut_ptr(),
//                 1,
//             );
//         } else if lhs.shape_stride().shape().len() == 1 {
//             // call gemv
//             <Self as MatrixMul<T, Dim, RHS, LHS>>::Blas::gemv(
//                 <Self as MatrixMul<T, Dim, RHS, LHS>>::Blas::Transpose::NoTrans,
//                 lhs.shape_stride().shape()[0],
//                 self.shape_stride().shape()[0],
//                 unsafe { *lhs.memory().as_ptr() },
//                 lhs.view_memory().as_ptr(),
//                 lhs.shape_stride().stride()[0],
//                 self.view_mut_memory().as_mut_ptr(),
//                 1,
//             );
//         } else if lhs.shape_stride().shape().len() == 2 {
//             // call gemm
//             <Self as MatrixMul<T, Dim, RHS, LHS>>::Blas::gemm(
//                 <Self as MatrixMul<T, Dim, RHS, LHS>>::Blas::Transpose::NoTrans,
//                 <Self as MatrixMul<T, Dim, RHS, LHS>>::Blas::Transpose::NoTrans,
//                 lhs.shape_stride().shape()[0],
//                 self.shape_stride().shape()[1],
//                 lhs.shape_stride().shape()[1],
//                 unsafe { *lhs.memory().as_ptr() },
//                 lhs.view_memory().as_ptr(),
//                 lhs.shape_stride().stride()[0],
//                 self.view_mut_memory().as_mut_ptr(),
//                 self.shape_stride().stride()[0],
//             );
//         } else {
//             unimplemented!()
//         }
//     }
// }

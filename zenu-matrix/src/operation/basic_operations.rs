use crate::{
    device::{cpu::Cpu, DeviceBase},
    dim::{larger_shape, smaller_shape, DimDyn, DimTrait},
    index::Index0D,
    matrix::{Matrix, Ref, Repr},
    num::Num,
};

fn get_tmp_matrix<R: Repr, S: DimTrait, D: DeviceBase>(
    a: &Matrix<R, S, D>,
    len: usize,
    idx: usize,
    self_len: usize,
) -> Matrix<Ref<&R::Item>, DimDyn, D> {
    if self_len == len {
        if a.shape()[0] == 1 {
            a.index_axis_dyn(Index0D::new(0))
        } else {
            a.index_axis_dyn(Index0D::new(idx))
        }
    } else {
        a.to_ref().into_dyn_dim()
    }
}

pub trait Add<T: Num>: DeviceBase {
    fn array_array(
        to: *mut T,
        lhs: *const T,
        rhs: *const T,
        num_elm: usize,
        to_stride: usize,
        lhs_stride: usize,
        rhs_stride: usize,
    );

    fn array_assign(to: *mut T, rhs: *const T, num_elm: usize, to_stride: usize, rhs_stride: usize);

    fn scalar(
        to: *mut T,
        lhs: *const T,
        rhs: T,
        num_elm: usize,
        to_stride: usize,
        lhs_stride: usize,
    );

    fn scalar_assign(to: *mut T, rhs: T, num_elm: usize, to_stride: usize);
}

impl<T: Num> Add<T> for Cpu {
    fn array_array(
        to: *mut T,
        lhs: *const T,
        rhs: *const T,
        num_elm: usize,
        to_stride: usize,
        lhs_stride: usize,
        rhs_stride: usize,
    ) {
        unsafe {
            for i in 0..num_elm {
                *to.add(i * to_stride) = *lhs.add(i * lhs_stride) + *rhs.add(i * rhs_stride);
            }
        }
    }

    fn array_assign(
        to: *mut T,
        rhs: *const T,
        num_elm: usize,
        to_stride: usize,
        rhs_stride: usize,
    ) {
        unsafe {
            for i in 0..num_elm {
                *to.add(i * to_stride) += *rhs.add(i * rhs_stride);
            }
        }
    }

    fn scalar(
        to: *mut T,
        lhs: *const T,
        rhs: T,
        num_elm: usize,
        to_stride: usize,
        lhs_stride: usize,
    ) {
        unsafe {
            for i in 0..num_elm {
                *to.add(i * to_stride) = *lhs.add(i * lhs_stride) + rhs;
            }
        }
    }

    fn scalar_assign(to: *mut T, rhs: T, num_elm: usize, to_stride: usize) {
        unsafe {
            for i in 0..num_elm {
                *to.add(i * to_stride) += rhs;
            }
        }
    }
}

/// 1dのMatrixを受け取る(これは入力側でチェック)
/// その配列の中身が1かどうかを確認
/// 1ならtrueを返す
fn is_1d_1(a: &[usize]) -> bool {
    a[0] == 1
}
// macro_rules! impl_basic_1d_functions_no_input {
//     (
//         $mod_name:ident,
//         $method:ident
//     ) => {
//         mod $mod_name {
//             use crate::{
//                 dim::DimTrait,
//                 matrix::MatrixBase,
//                 matrix_impl::Matrix,
//                 memory_impl::{ViewMem, ViewMutMem},
//                 num::Num,
//             };
//
//             pub fn _1d_1d_cpu<T: Num, D1: DimTrait, D2: DimTrait>(
//                 to: &mut Matrix<ViewMutMem<T>, D1>,
//                 a: &Matrix<ViewMem<T>, D2>,
//             ) {
//                 let num_elm = to.shape().num_elm();
//                 let to_stride = to.stride()[0];
//                 let a_stride = a.stride()[0];
//                 let slice_to = to.as_mut_slice();
//                 let slice_a = a.as_slice();
//                 for i in 0..num_elm {
//                     slice_to[i * to_stride] = slice_a[i * a_stride].$method();
//                 }
//             }
//         }
//     };
// }
// macro_rules! impl_traits_no_input {
//     (
//         $trait:ident,
//         $trait_method:ident,
//         $mod_name:ident,
//         $method:ident
//     ) => {
//         pub trait $trait<L> {
//             fn $trait_method(&mut self, lhs: L);
//         }
//
//         impl<T, D1, D3, M1, M3> $trait<Matrix<M1, D1>> for Matrix<M3, D3>
//         where
//             T: Num,
//             D1: DimTrait,
//             D3: DimTrait,
//             M1: ToViewMemory<Item = T>,
//             M3: ToViewMutMemory<Item = T>,
//         {
//             fn $trait_method(&mut self, lhs: Matrix<M1, D1>) {
//                 assert_eq!(
//                     self.shape().slice(),
//                     lhs.shape().slice(),
//                     "Matrix shape mismatch"
//                 );
//                 if self.shape().is_empty() {
//                     let mut view_mut = self.to_view_mut();
//                     let self_slice = view_mut.as_mut_slice();
//                     let lhs_slice = lhs.as_slice();
//                     self_slice[0] = lhs_slice[0].$method();
//                 } else if self.shape().len() == 1 {
//                     $mod_name::_1d_1d_cpu(&mut self.to_view_mut(), &lhs.to_view());
//                 } else {
//                     let num_iter = self.shape()[0];
//                     let self_dim_len = self.shape().len();
//                     let lhs_dim_len = lhs.shape().len();
//                     for idx in 0..num_iter {
//                         let mut s = self.index_axis_mut_dyn(Index0D::new(idx));
//                         let lhs = get_tmp_matrix(&lhs, lhs_dim_len, idx, self_dim_len);
//                         s.$trait_method(lhs);
//                     }
//                 }
//             }
//         }
//     };
// }
// impl_basic_1d_functions_no_input!(sin_mod, sin);
// impl_traits_no_input!(MatrixSin, sin, sin_mod, sin);
// impl_basic_1d_functions_no_input!(cos_mod, cos);
// impl_traits_no_input!(MatrixCos, cos, cos_mod, cos);
// impl_basic_1d_functions_no_input!(tan_mod, tan);
// impl_traits_no_input!(MatrixTan, tan, tan_mod, tan);
// impl_basic_1d_functions_no_input!(asin_mod, asin);
// impl_traits_no_input!(MatrixAsin, asin, asin_mod, asin);
// impl_basic_1d_functions_no_input!(acos_mod, acos);
// impl_traits_no_input!(MatrixAcos, acos, acos_mod, acos);
// impl_basic_1d_functions_no_input!(atan_mod, atan);
// impl_traits_no_input!(MatrixAtan, atan, atan_mod, atan);
// impl_basic_1d_functions_no_input!(sinh_mod, sinh);
// impl_traits_no_input!(MatrixSinh, sinh, sinh_mod, sinh);
// impl_basic_1d_functions_no_input!(cosh_mod, cosh);
// impl_traits_no_input!(MatrixCosh, cosh, cosh_mod, cosh);
// impl_basic_1d_functions_no_input!(tanh_mod, tanh);
// impl_traits_no_input!(MatrixTanh, tanh, tanh_mod, tanh);
// impl_basic_1d_functions_no_input!(asinh_mod, asinh);
// impl_traits_no_input!(MatrixAsinh, asinh, asinh_mod, asinh);
// impl_basic_1d_functions_no_input!(acosh_mod, acosh);
// impl_traits_no_input!(MatrixAcosh, acosh, acosh_mod, acosh);
// impl_basic_1d_functions_no_input!(atanh_mod, atanh);
// impl_traits_no_input!(MatrixAtanh, atanh, atanh_mod, atanh);
// impl_basic_1d_functions_no_input!(sqrt_mod, sqrt);
// impl_traits_no_input!(MatrixSqrt, sqrt, sqrt_mod, sqrt);
// impl_basic_1d_functions_no_input!(abs_mod, abs);
// impl_traits_no_input!(MatrixAbs, abs, abs_mod, abs);
//
// macro_rules! impl_basic_1d_functions {
//     (
//         $mod_name:ident,
//         $method:ident,
//         $($assign_method:ident)?
//     ) => {
//         mod $mod_name {
//             use crate::{
//                 dim::DimTrait,
//                 matrix::MatrixBase,
//                 matrix_impl::Matrix,
//                 memory_impl::{ViewMem, ViewMutMem},
//                 num::Num,
//             };
//
//             pub fn _1d_1d_cpu<T: Num, D1: DimTrait, D2: DimTrait, D3: DimTrait>(
//                 to: &mut Matrix<ViewMutMem<T>, D1>,
//                 a: &Matrix<ViewMem<T>, D2>,
//                 b: &Matrix<ViewMem<T>, D3>,
//             ) {
//                 let num_elm = to.shape().num_elm();
//                 let to_stride = to.stride()[0];
//                 let a_stride = a.stride()[0];
//                 let b_stride = b.stride()[0];
//                 let slice_to = to.as_mut_slice();
//                 let slice_a = a.as_slice();
//                 let slice_b = b.as_slice();
//                 for i in 0..num_elm {
//                     slice_to[i * to_stride] = slice_a[i * a_stride].$method(slice_b[i * b_stride]);
//                 }
//             }
//
//             pub fn _1d_scalar_cpu<T: Num, D1: DimTrait, D2: DimTrait>(
//                 to: &mut Matrix<ViewMutMem<T>, D1>,
//                 a: &Matrix<ViewMem<T>, D2>,
//                 b: T,
//             ) {
//                 let num_elm = to.shape().num_elm();
//                 let to_stride = to.stride()[0];
//                 let a_stride = a.stride()[0];
//                 let slice_to = to.as_mut_slice();
//                 let slice_a = a.as_slice();
//                 for i in 0..num_elm {
//                     slice_to[i * to_stride] = slice_a[i * a_stride].$method(b);
//                 }
//             }
//
//             pub fn _scalar_1d_cpu<T: Num, D1: DimTrait, D2: DimTrait>(
//                 to: &mut Matrix<ViewMutMem<T>, D1>,
//                 b: T,
//                 a: &Matrix<ViewMem<T>, D2>,
//             ) {
//                 let num_elm = to.shape().num_elm();
//                 let to_stride = to.stride()[0];
//                 let a_stride = a.stride()[0];
//                 let slice_to = to.as_mut_slice();
//                 let slice_a = a.as_slice();
//                 for i in 0..num_elm {
//                     slice_to[i * to_stride] = slice_a[i * a_stride].$method(b);
//                 }
//             }
//         $(
//             pub fn assign_1d_1d_cpu<T: Num, D1: DimTrait, D2: DimTrait>(
//                 to: &mut Matrix<ViewMutMem<T>, D1>,
//                 b: &Matrix<ViewMem<T>, D2>,
//             ) {
//                 let num_elm = to.shape().num_elm();
//                 let to_stride = to.stride()[0];
//                 let b_stride = b.stride()[0];
//                 let slice_to = to.as_mut_slice();
//                 let slice_b = b.as_slice();
//                 for i in 0..num_elm {
//                     slice_to[i * to_stride].$assign_method(slice_b[i * b_stride]);
//                 }
//             }
//
//             pub fn assign_1d_scalar_cpu<T: Num, D: DimTrait>(
//                 to: &mut Matrix<ViewMutMem<T>, D>,
//                 b: T,
//             ) {
//                 let num_elm = to.shape().num_elm();
//                 let to_stride = to.stride()[0];
//                 let slice_to = to.as_mut_slice();
//                 for i in 0..num_elm {
//                     slice_to[i * to_stride].$assign_method(b);
//                 }
//             }
//         )?
//         }
//     };
// }
macro_rules! impl_basic_ops {
    (
        $method:ident,
        $assign_method:ident,
        $scalar_method:ident,
        $scalar_assign_method:ident,
        $device_trait:ident
    ) => {
        impl<'a, T, S, D> Matrix<Ref<&mut T>, S, D>
        where
            T: Num,
            S: DimTrait,
            D: DeviceBase + $device_trait<T>,
        {
            pub fn $scalar_method<RL: Repr<Item=T>, SL: DimTrait>(&mut self, lhs: Matrix<RL, SL, D>, rhs: T) {
                if self.shape().slice() != lhs.shape().slice() {
                    panic!("Matrix shape mismatch");
                }

                if self.shape().is_empty() {
                    let self_slice = self.as_mut_slice();
                    let lhs_slice = lhs.as_slice();
                    self_slice[0] = lhs_slice[0].$method(rhs);
                } else if self.shape().len() == 1 {
                    D::scalar(
                        self.as_mut_ptr(),
                        lhs.as_ptr(),
                        rhs,
                        self.shape().num_elm(),
                        self.stride()[0],
                        lhs.stride()[0]
                    );
                } else {
                    let num_iter = self.shape()[0];
                    for idx in 0..num_iter {
                        let mut s = self.index_axis_mut_dyn(Index0D::new(idx));
                        let lhs = lhs.index_axis_dyn(Index0D::new(idx));
                        s.$scalar_method(lhs, rhs);
                    }
                }
            }

            pub fn $scalar_assign_method(&mut self, rhs: T) {
                if self.shape().is_empty() {
                    D::scalar_assign(
                        self.as_mut_ptr(),
                        rhs,
                        self.shape().num_elm(),
                        self.stride()[0]
                    );
                } else if self.shape().len() == 1 {
                    D::scalar_assign(self.as_mut_ptr(), rhs, self.shape().num_elm(), self.stride()[0]);
                } else {
                    let num_iter = self.shape()[0];
                    for idx in 0..num_iter {
                        let mut s = self.index_axis_mut_dyn(Index0D::new(idx));
                        s.$scalar_assign_method(rhs);
                    }
                }
            }

            pub fn $method<RL, RR, SL, SR>(&mut self, lhs: Matrix<RL, SL, D>, rhs: Matrix<RR, SR, D>)
            where
                RL: Repr<Item=T>,
                RR: Repr<Item=T>,
                SL: DimTrait,
                SR: DimTrait,
            {
                let larger_dim = larger_shape(lhs.shape(), rhs.shape());
                let smaller_dim = smaller_shape(lhs.shape(), rhs.shape());

                if !(larger_dim.is_include(smaller_dim) || DimDyn::from(self.shape().slice()).is_include_bradcast(smaller_dim)) {
                    panic!(
                        "self dim is not match other dims self dim {:?}, lhs dim {:?} rhs dim {:?}",
                        self.shape(),
                        lhs.shape(),
                        rhs.shape()
                    );
                }
                if self.shape().slice() != larger_dim.slice() && self.shape().slice() != smaller_dim.slice() {
                    panic!("longer shape lhs or rhs is same shape to self\n self.shape = {:?}\n lhs.shape() = {:?} \n rhs.shape() = {:?}", self.shape(), lhs.shape(), rhs.shape());
                }

                if rhs.shape().is_empty() {
                    self.$scalar_method(lhs, rhs.as_slice()[0]);
                    return;
                }

                if lhs.shape().is_empty() {
                    self.$scalar_method(rhs, lhs.index_item(&[] as &[usize]));
                    return;
                }

                if self.shape().is_empty() {
                    let self_slice = self.as_mut_slice();
                    let lhs_slice = lhs.as_slice();
                    let rhs_slice = rhs.as_slice();
                    self_slice[0] = lhs_slice[0].$method(rhs_slice[0]);
                } else if self.shape().len() == 1 {
                    if is_1d_1(lhs.shape().slice()) {
                        D::scalar(
                            self.as_mut_ptr(),
                            rhs.as_ptr(),
                            lhs.index_item(&[] as &[usize]),
                            self.shape().num_elm(),
                            self.stride()[0],
                            rhs.stride()[0]
                        );
                        return;
                    } else if is_1d_1(rhs.shape().slice()) {
                        D::scalar(
                            self.as_mut_ptr(),
                            lhs.as_ptr(),
                            unsafe { *rhs.as_ptr() },
                            self.shape().num_elm(),
                            self.stride()[0],
                            lhs.stride()[0]
                        );
                        return;
                    }

                    D::array_array(
                        self.as_mut_ptr(),
                        lhs.as_ptr(),
                        rhs.as_ptr(),
                        self.shape().num_elm(),
                        self.stride()[0],
                        lhs.stride()[0],
                        rhs.stride()[0]
                    );
                } else {
                    let num_iter = self.shape()[0];
                    let self_dim_len = self.shape().len();
                    let rhs_dim_len = rhs.shape().len();
                    let lhs_dim_len = lhs.shape().len();
                    for idx in 0..num_iter {
                        let mut s = self.index_axis_mut_dyn(Index0D::new(idx));
                        let lhs = get_tmp_matrix(&lhs, lhs_dim_len, idx, self_dim_len);
                        let rhs = get_tmp_matrix(&rhs, rhs_dim_len, idx, self_dim_len);
                        s.$method(lhs, rhs);
                    }
                }
            }

            pub fn $assign_method<RR, SR>(&mut self, rhs: Matrix<RR, SR, D>)
            where
                RR: Repr<Item=T>,
                SR: DimTrait,
            {
                if self.shape().len() < rhs.shape().len() {
                    panic!("Self shape len is larger than rhs shape len {:?} {:?}", self.shape(), rhs.shape());
                }

                if !(DimDyn::from(self.shape().slice())
                    .is_include(DimDyn::from(rhs.shape().slice()))
                    || DimDyn::from(self.shape().slice())
                    .is_include_bradcast(DimDyn::from(rhs.shape().slice()))
                )
                {
                    panic!("rhs shape is not include self shape {:?} {:?}", self.shape(), rhs.shape());
                }

                if !DimDyn::from(self.shape().slice())
                    .is_include_bradcast(DimDyn::from(rhs.shape().slice()))
                {
                    panic!("rhs shape is not include self shape {:?} {:?}", self.shape(), rhs.shape());
                }

                if self.shape().is_empty() {
                    let self_slice = self.as_mut_slice();
                    let rhs_slice = rhs.as_slice();
                    self_slice[0].$assign_method(rhs_slice[0]);
                } else if rhs.shape().is_empty() {
                    self.$scalar_assign_method(rhs.index_item(&[] as &[usize]));
                } else if self.shape().len() == 1 {
                    if is_1d_1(rhs.shape().slice()) {
                        self.$scalar_assign_method(
                            rhs.index_item(&[] as &[usize])
                        );
                        return;
                    }
                    D::array_assign(
                        self.as_mut_ptr(),
                        rhs.as_ptr(),
                        self.shape().num_elm(),
                        self.stride()[0],
                        rhs.stride()[0]
                    );
                } else {
                    let num_iter = self.shape()[0];
                    let self_shape_len = self.shape().len();
                    let rhs_shape_len = rhs.shape().len();
                    for idx in 0..num_iter {
                        let mut s = self.index_axis_mut_dyn(Index0D::new(idx));
                        let rhs = get_tmp_matrix(&rhs, rhs_shape_len, idx, self_shape_len);
                        s.$assign_method(rhs);
                    }
                }
            }

        }
    };
}
// impl_basic_1d_functions!(add_mod, add, add_assign);
impl_basic_ops!(add, add_assign, add_scalar, add_scalar_assign, Add);
// impl_basic_1d_functions!(sub_mod, sub, sub_assign);
// impl_traits!(
//     MatrixSub,
//     sub,
//     MatrixSubAssign,
//     sub_assign,
//     sub_mod,
//     sub,
//     sub_assign
// );
// impl_basic_1d_functions!(mul_mod, mul, mul_assign);
// impl_traits!(
//     MatrixMul,
//     mul,
//     MatrixMulAssign,
//     mul_assign,
//     mul_mod,
//     mul,
//     mul_assign
// );
// impl_basic_1d_functions!(div_mod, div, div_assign);
// impl_traits!(
//     MatrixDiv,
//     div,
//     MatrixDivAssign,
//     div_assign,
//     div_mod,
//     div,
//     div_assign
// );
// impl_basic_1d_functions!(powf_mod, powf,);
// impl_traits!(
//     MatrixPowf,
//     powf,
//     MatrixPowfAssign,
//     powf_assign,
//     powf_mod,
//     powf,
// );
// impl_basic_1d_functions!(log_mod, log,);
// impl_traits!(MatrixLog, log, MatrixLogAssign, log_assign, log_mod, log,);
//
// #[cfg(test)]
// mod add {
//     use crate::{
//         constructor::zeros::Zeros,
//         matrix::{IndexItem, MatrixSlice, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
//         matrix_impl::{OwnedMatrix0D, OwnedMatrix1D, OwnedMatrix2D, OwnedMatrix3D, OwnedMatrixDyn},
//         operation::asum::Asum,
//         slice,
//     };
//
//     use super::*;
//
//     #[test]
//     fn add_0d_0d() {
//         let a = OwnedMatrix0D::from_vec(vec![1.0], []);
//         let b = OwnedMatrix0D::from_vec(vec![1.0], []);
//         let mut ans = OwnedMatrix0D::<f32>::zeros([]);
//         ans.add(a.to_view(), b.to_view());
//         assert_eq!(ans.index_item([]), 2.0);
//     }
//
//     #[test]
//     fn add_dyn_dyn() {
//         let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
//         let b = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
//         let ans = OwnedMatrix1D::<f32>::zeros([3]);
//
//         let a = a.into_dyn_dim();
//         let b = b.into_dyn_dim();
//         let mut ans = ans.into_dyn_dim();
//
//         ans.to_view_mut().add(a.to_view(), b.to_view());
//     }
//
//     #[test]
//     fn add_1d_scalar() {
//         let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
//         let mut ans = OwnedMatrix1D::<f32>::zeros([3]);
//         let b = OwnedMatrix0D::from_vec(vec![2.0], []);
//         ans.to_view_mut().add(a.to_view(), b.to_view());
//
//         assert_eq!(ans.index_item([0]), 3.0);
//         assert_eq!(ans.index_item([1]), 4.0);
//         assert_eq!(ans.index_item([2]), 5.0);
//     }
//
//     #[test]
//     fn add_1d_scalar_default_stride() {
//         let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
//         let mut ans = OwnedMatrix1D::<f32>::zeros([3]);
//         ans.to_view_mut().add(a.to_view(), 1.0);
//
//         assert_eq!(ans.index_item([0]), 2.0);
//         assert_eq!(ans.index_item([1]), 3.0);
//         assert_eq!(ans.index_item([2]), 4.0);
//     }
//
//     #[test]
//     fn add_1d_scalar_sliced() {
//         let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
//         let mut ans = OwnedMatrix1D::<f32>::zeros([3]);
//
//         let sliced = a.slice(slice!(..;2));
//
//         ans.to_view_mut().add(sliced.to_view(), 1.0);
//
//         assert_eq!(ans.index_item([0]), 2.0);
//         assert_eq!(ans.index_item([1]), 4.0);
//         assert_eq!(ans.index_item([2]), 6.0);
//     }
//
//     #[test]
//     fn add_3d_scalar_sliced() {
//         let a = OwnedMatrix3D::from_vec(
//             vec![
//                 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
//                 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
//                 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
//             ],
//             [3, 3, 4],
//         );
//
//         let mut ans = OwnedMatrix3D::<f32>::zeros([3, 3, 2]);
//
//         let sliced = a.slice(slice!(.., .., ..;2));
//
//         ans.to_view_mut().add(sliced.to_view(), 1.0);
//
//         assert_eq!(ans.index_item([0, 0, 0]), 2.0);
//         assert_eq!(ans.index_item([0, 0, 1]), 4.0);
//         assert_eq!(ans.index_item([0, 1, 0]), 6.0);
//         assert_eq!(ans.index_item([0, 1, 1]), 8.0);
//         assert_eq!(ans.index_item([0, 2, 0]), 10.0);
//         assert_eq!(ans.index_item([0, 2, 1]), 12.0);
//         assert_eq!(ans.index_item([1, 0, 0]), 14.0);
//         assert_eq!(ans.index_item([1, 0, 1]), 16.0);
//         assert_eq!(ans.index_item([1, 1, 0]), 18.0);
//         assert_eq!(ans.index_item([1, 1, 1]), 20.0);
//         assert_eq!(ans.index_item([1, 2, 0]), 22.0);
//         assert_eq!(ans.index_item([1, 2, 1]), 24.0);
//         assert_eq!(ans.index_item([2, 0, 0]), 26.0);
//         assert_eq!(ans.index_item([2, 0, 1]), 28.0);
//         assert_eq!(ans.index_item([2, 1, 0]), 30.0);
//         assert_eq!(ans.index_item([2, 1, 1]), 32.0);
//         assert_eq!(ans.index_item([2, 2, 0]), 34.0);
//         assert_eq!(ans.index_item([2, 2, 1]), 36.0);
//     }
//
//     #[test]
//     fn add_1d_1d_default_stride() {
//         let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
//         let b = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
//         let mut ans = OwnedMatrix1D::<f32>::zeros([3]);
//         ans.to_view_mut().add(a.to_view(), b.to_view());
//
//         assert_eq!(ans.index_item([0]), 2.0);
//         assert_eq!(ans.index_item([1]), 4.0);
//         assert_eq!(ans.index_item([2]), 6.0);
//     }
//
//     #[test]
//     fn add_1d_1d_sliced() {
//         let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
//         let b = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
//
//         let mut ans = OwnedMatrix1D::<f32>::zeros([3]);
//
//         let sliced_a = a.slice(slice!(..;2));
//         let sliced_b = b.slice(slice!(1..;2));
//
//         ans.to_view_mut()
//             .add(sliced_a.to_view(), sliced_b.to_view());
//
//         assert_eq!(ans.index_item([0]), 3.0);
//         assert_eq!(ans.index_item([1]), 7.0);
//         assert_eq!(ans.index_item([2]), 11.0);
//     }
//
//     #[test]
//     fn add_2d_1d_default() {
//         let a = OwnedMatrix2D::from_vec(
//             vec![
//                 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
//             ],
//             [4, 4],
//         );
//
//         let b = OwnedMatrix1D::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], [8]);
//
//         let mut ans = OwnedMatrix2D::<f32>::zeros([2, 2]);
//
//         let sliced_a = a.slice(slice!(..2, ..2));
//         let sliced_b = b.slice(slice!(..2));
//
//         ans.to_view_mut()
//             .add(sliced_a.to_view(), sliced_b.to_view());
//
//         assert_eq!(ans.index_item([0, 0]), 2.0);
//         assert_eq!(ans.index_item([0, 1]), 4.0);
//         assert_eq!(ans.index_item([1, 0]), 6.0);
//         assert_eq!(ans.index_item([1, 1]), 8.0);
//     }
//
//     #[test]
//     fn add_3d_1d_sliced() {
//         let mut v = Vec::new();
//         let num_elm = 4 * 4 * 4;
//         for i in 0..num_elm {
//             v.push(i as f32);
//         }
//         let a = OwnedMatrix3D::from_vec(v, [4, 4, 4]);
//
//         let b = OwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], [4]);
//
//         let mut ans = OwnedMatrix3D::<f32>::zeros([2, 2, 2]);
//
//         let sliced_a = a.slice(slice!(..2, 1..;2, ..2));
//         let sliced_b = b.slice(slice!(..2));
//
//         ans.to_view_mut()
//             .add(sliced_a.to_view(), sliced_b.to_view());
//
//         assert_eq!(ans.index_item([0, 0, 0]), 5.);
//         assert_eq!(ans.index_item([0, 0, 1]), 7.);
//         assert_eq!(ans.index_item([0, 1, 0]), 13.);
//         assert_eq!(ans.index_item([0, 1, 1]), 15.);
//         assert_eq!(ans.index_item([1, 0, 0]), 21.);
//         assert_eq!(ans.index_item([1, 0, 1]), 23.);
//         assert_eq!(ans.index_item([1, 1, 0]), 29.);
//         assert_eq!(ans.index_item([1, 1, 1]), 31.);
//     }
//
//     #[test]
//     fn add_2d_2d_default() {
//         let a = OwnedMatrix2D::from_vec(
//             vec![
//                 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
//             ],
//             [4, 4],
//         );
//
//         let b = OwnedMatrix2D::from_vec(
//             vec![
//                 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
//             ],
//             [4, 4],
//         );
//
//         let mut ans = OwnedMatrix2D::<f32>::zeros([4, 4]);
//         ans.to_view_mut().add(a.to_view(), b.to_view());
//
//         assert_eq!(ans.index_item([0, 0]), 2.0);
//         assert_eq!(ans.index_item([0, 1]), 4.0);
//         assert_eq!(ans.index_item([0, 2]), 6.0);
//         assert_eq!(ans.index_item([0, 3]), 8.0);
//         assert_eq!(ans.index_item([1, 0]), 10.0);
//         assert_eq!(ans.index_item([1, 1]), 12.0);
//         assert_eq!(ans.index_item([1, 2]), 14.0);
//         assert_eq!(ans.index_item([1, 3]), 16.0);
//         assert_eq!(ans.index_item([2, 0]), 18.0);
//         assert_eq!(ans.index_item([2, 1]), 20.0);
//         assert_eq!(ans.index_item([2, 2]), 22.0);
//         assert_eq!(ans.index_item([2, 3]), 24.0);
//         assert_eq!(ans.index_item([3, 0]), 26.0);
//         assert_eq!(ans.index_item([3, 1]), 28.0);
//         assert_eq!(ans.index_item([3, 2]), 30.0);
//         assert_eq!(ans.index_item([3, 3]), 32.0);
//     }
//
//     #[test]
//     fn add_2d_0d() {
//         let a = OwnedMatrix2D::from_vec(
//             vec![
//                 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
//             ],
//             [4, 4],
//         );
//         let b = OwnedMatrix0D::from_vec(vec![1.], []);
//         let mut ans = OwnedMatrix2D::<f32>::zeros([4, 4]);
//         ans.to_view_mut().add(a.to_view(), b.to_view());
//         assert_eq!(ans.index_item([0, 0]), 2.0);
//         assert_eq!(ans.index_item([0, 1]), 3.0);
//         assert_eq!(ans.index_item([0, 2]), 4.0);
//         assert_eq!(ans.index_item([0, 3]), 5.0);
//         assert_eq!(ans.index_item([1, 0]), 6.0);
//         assert_eq!(ans.index_item([1, 1]), 7.0);
//         assert_eq!(ans.index_item([1, 2]), 8.0);
//         assert_eq!(ans.index_item([1, 3]), 9.0);
//         assert_eq!(ans.index_item([2, 0]), 10.0);
//         assert_eq!(ans.index_item([2, 1]), 11.0);
//         assert_eq!(ans.index_item([2, 2]), 12.0);
//         assert_eq!(ans.index_item([2, 3]), 13.0);
//         assert_eq!(ans.index_item([3, 0]), 14.0);
//         assert_eq!(ans.index_item([3, 1]), 15.0);
//         assert_eq!(ans.index_item([3, 2]), 16.0);
//         assert_eq!(ans.index_item([3, 3]), 17.0);
//     }
//
//     #[test]
//     fn add_2d_0d_dyn() {
//         let a = OwnedMatrixDyn::from_vec(
//             vec![
//                 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
//             ],
//             [4, 4],
//         );
//         let b = OwnedMatrixDyn::from_vec(vec![1.], []);
//         let mut ans = OwnedMatrixDyn::<f32>::zeros([4, 4]);
//         ans.to_view_mut().add(a.to_view(), b.to_view());
//         assert_eq!(ans.index_item([0, 0]), 2.0);
//         assert_eq!(ans.index_item([0, 1]), 3.0);
//         assert_eq!(ans.index_item([0, 2]), 4.0);
//         assert_eq!(ans.index_item([0, 3]), 5.0);
//         assert_eq!(ans.index_item([1, 0]), 6.0);
//         assert_eq!(ans.index_item([1, 1]), 7.0);
//         assert_eq!(ans.index_item([1, 2]), 8.0);
//         assert_eq!(ans.index_item([1, 3]), 9.0);
//         assert_eq!(ans.index_item([2, 0]), 10.0);
//         assert_eq!(ans.index_item([2, 1]), 11.0);
//         assert_eq!(ans.index_item([2, 2]), 12.0);
//         assert_eq!(ans.index_item([2, 3]), 13.0);
//         assert_eq!(ans.index_item([3, 0]), 14.0);
//         assert_eq!(ans.index_item([3, 1]), 15.0);
//         assert_eq!(ans.index_item([3, 2]), 16.0);
//         assert_eq!(ans.index_item([3, 3]), 17.0);
//     }
//
//     #[test]
//     fn add_4d_2d_dyn() {
//         let zeros_4d = OwnedMatrixDyn::<f32>::zeros([2, 2, 2, 2]);
//         let ones_2d = OwnedMatrixDyn::from_vec(vec![1., 1., 1., 1.], [2, 2]);
//         let mut ans = OwnedMatrixDyn::<f32>::zeros([2, 2, 2, 2]);
//         ans.to_view_mut().add(zeros_4d.to_view(), ones_2d.to_view());
//     }
//
//     #[test]
//     fn broad_cast_4x1x1x1_4x3x3x3() {
//         let a = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4.], [4, 1, 1, 1]);
//         let b = OwnedMatrixDyn::zeros([4, 2, 3, 3]);
//         let mut ans = OwnedMatrixDyn::<f32>::zeros([4, 2, 3, 3]);
//         ans.to_view_mut().add(a.to_view(), b.to_view());
//         let one = vec![1; 2 * 3 * 3];
//         let two = vec![2; 2 * 3 * 3];
//         let three = vec![3; 2 * 3 * 3];
//         let four = vec![4; 2 * 3 * 3];
//         let mut result = Vec::new();
//         result.extend_from_slice(&one);
//         result.extend_from_slice(&two);
//         result.extend_from_slice(&three);
//         result.extend_from_slice(&four);
//         let result = result.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
//         let result = OwnedMatrixDyn::from_vec(result, [4, 2, 3, 3]);
//         assert!((ans.to_view() - result.to_view()).asum() == 0.0);
//     }
// }
//
// #[cfg(test)]
// mod sub {
//     use crate::{
//         constructor::zeros::Zeros,
//         matrix::{IndexItem, OwnedMatrix},
//         matrix_impl::OwnedMatrix0D,
//         operation::basic_operations::MatrixSub,
//     };
//
//     #[test]
//     fn sub_0d_0d() {
//         let a = OwnedMatrix0D::from_vec(vec![1.0], []);
//         let b = OwnedMatrix0D::from_vec(vec![1.0], []);
//         let mut ans = OwnedMatrix0D::<f32>::zeros([]);
//         ans.sub(a, b);
//         assert_eq!(ans.index_item([]), 0.0);
//     }
// }
//
// #[cfg(test)]
// mod div {
//     use crate::{
//         constructor::zeros::Zeros,
//         matrix::{OwnedMatrix, ToViewMatrix},
//         matrix_impl::OwnedMatrixDyn,
//         operation::{
//             asum::Asum,
//             basic_operations::{MatrixDiv, MatrixDivAssign},
//         },
//     };
//
//     #[test]
//     fn div_0d_0d() {
//         let mut a = OwnedMatrixDyn::from_vec(vec![2.0], &[]);
//         let b = OwnedMatrixDyn::from_vec(vec![3.0], &[]);
//         let mut c = OwnedMatrixDyn::zeros_like(a.to_view());
//
//         c.div(a.to_view(), b.to_view());
//         assert_eq!(c.as_slice(), &[2.0 / 3.0]);
//
//         a.div_assign(b);
//         assert_eq!(a.as_slice(), &[2.0 / 3.0]);
//     }
//
//     #[test]
//     fn div_1d_0d() {
//         let mut a = OwnedMatrixDyn::from_vec(vec![2.0, 3.0], &[2]);
//         let b = OwnedMatrixDyn::from_vec(vec![3.0], &[]);
//         let mut c = OwnedMatrixDyn::zeros_like(a.to_view());
//         c.div(a.to_view(), b.to_view());
//         assert_eq!(c.as_slice(), &[2.0 / 3.0, 3.0 / 3.0]);
//
//         a.div_assign(b);
//         assert_eq!(a.as_slice(), &[2.0 / 3.0, 3.0 / 3.0]);
//     }
//
//     #[test]
//     fn div_3d_3d() {
//         let mut a =
//             OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
//         let b = OwnedMatrixDyn::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[2, 2, 2]);
//         let mut c = OwnedMatrixDyn::zeros_like(a.to_view());
//         c.div(a.to_view(), b.to_view());
//         let ans = vec![
//             1.0 / 2.0,
//             2.0 / 3.0,
//             3.0 / 4.0,
//             4.0 / 5.0,
//             5.0 / 6.0,
//             6.0 / 7.0,
//             7.0 / 8.0,
//             8.0 / 9.0,
//         ];
//         let ans = OwnedMatrixDyn::from_vec(ans, &[2, 2, 2]);
//         let diff = c.to_view() - ans.to_view();
//         let asum = diff.asum();
//         assert_eq!(asum, 0.0);
//
//         a.div_assign(b);
//         let diff = a.to_view() - ans.to_view();
//         let asum = diff.asum();
//         assert_eq!(asum, 0.0);
//     }
//
//     #[test]
//     fn div_4d_2d() {
//         let mut a = OwnedMatrixDyn::from_vec(
//             vec![
//                 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
//             ],
//             &[2, 2, 2, 2],
//         );
//         let b = OwnedMatrixDyn::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
//         let mut c = OwnedMatrixDyn::zeros_like(a.to_view());
//         c.div(a.to_view(), b.to_view());
//         let ans = vec![
//             1.0 / 2.0,
//             2.0 / 3.0,
//             3.0 / 4.0,
//             4.0 / 5.0,
//             5.0 / 2.0,
//             6.0 / 3.0,
//             7.0 / 4.0,
//             8.0 / 5.0,
//             1.0 / 2.0,
//             2.0 / 3.0,
//             3.0 / 4.0,
//             4.0 / 5.0,
//             5.0 / 2.0,
//             6.0 / 3.0,
//             7.0 / 4.0,
//             8.0 / 5.0,
//         ];
//         let ans = OwnedMatrixDyn::from_vec(ans, &[2, 2, 2, 2]);
//         let diff = c.to_view() - ans.to_view();
//         let asum = diff.asum();
//         assert_eq!(asum, 0.0);
//
//         a.div_assign(b);
//         let diff = a.to_view() - ans.to_view();
//         let asum = diff.asum();
//         assert_eq!(asum, 0.0);
//     }
//
//     #[test]
//     fn broadcast_4d_4d() {
//         let a = OwnedMatrixDyn::from_vec(
//             vec![
//                 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
//             ],
//             &[2, 2, 2, 2],
//         );
//         let b = OwnedMatrixDyn::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[1, 1, 2, 2]);
//         let mut c = OwnedMatrixDyn::zeros_like(a.to_view());
//         c.div(a.to_view(), b.to_view());
//         let ans = vec![
//             1.0 / 2.0,
//             2.0 / 3.0,
//             3.0 / 4.0,
//             4.0 / 5.0,
//             5.0 / 2.0,
//             6.0 / 3.0,
//             7.0 / 4.0,
//             8.0 / 5.0,
//             1.0 / 2.0,
//             2.0 / 3.0,
//             3.0 / 4.0,
//             4.0 / 5.0,
//             5.0 / 2.0,
//             6.0 / 3.0,
//             7.0 / 4.0,
//             8.0 / 5.0,
//         ];
//         let ans = OwnedMatrixDyn::from_vec(ans, &[2, 2, 2, 2]);
//         let diff = c.to_view() - ans.to_view();
//         let asum = diff.asum();
//         assert_eq!(asum, 0.0);
//
//         c.div(b.to_view(), a.to_view());
//         let ans = vec![
//             2.0 / 1.0,
//             3.0 / 2.0,
//             4.0 / 3.0,
//             5.0 / 4.0,
//             2.0 / 5.0,
//             3.0 / 6.0,
//             4.0 / 7.0,
//             5.0 / 8.0,
//             2.0 / 1.0,
//             3.0 / 2.0,
//             4.0 / 3.0,
//             5.0 / 4.0,
//             2.0 / 5.0,
//             3.0 / 6.0,
//             4.0 / 7.0,
//             5.0 / 8.0,
//         ];
//         let ans = OwnedMatrixDyn::from_vec(ans, &[2, 2, 2, 2]);
//         let diff = c.to_view() - ans.to_view();
//         let asum = diff.asum();
//         assert_eq!(asum, 0.0);
//     }
// }
// #[cfg(test)]
// mod mul {
//     use crate::{
//         constructor::{ones::Ones, zeros::Zeros},
//         matrix::{IndexItem, MatrixSlice, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
//         matrix_impl::{OwnedMatrix0D, OwnedMatrix1D, OwnedMatrix2D, OwnedMatrix4D, OwnedMatrixDyn},
//         operation::basic_operations::MatrixMul,
//         slice,
//     };
//
//     use super::MatrixSin;
//
//     #[test]
//     fn mul_1d_scalar() {
//         let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
//         let b = OwnedMatrix0D::from_vec(vec![2.0], []);
//         let mut ans = OwnedMatrix1D::<f32>::zeros([3]);
//         ans.to_view_mut().mul(a.to_view(), b.to_view());
//
//         assert_eq!(ans.index_item([0]), 2.0);
//         assert_eq!(ans.index_item([1]), 4.0);
//         assert_eq!(ans.index_item([2]), 6.0);
//     }
//
//     #[test]
//     fn scalar_1d() {
//         let a = OwnedMatrix1D::from_vec(vec![1., 2., 3.], [3]);
//         let mut ans = OwnedMatrix1D::<f32>::zeros([3]);
//         ans.to_view_mut().mul(a.to_view(), 2.);
//
//         assert_eq!(ans.index_item([0]), 2.);
//         assert_eq!(ans.index_item([1]), 4.);
//         assert_eq!(ans.index_item([2]), 6.);
//     }
//
//     #[test]
//     fn sliced_scalar_1d() {
//         let a = OwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], [4]);
//         let mut ans = OwnedMatrix1D::<f32>::zeros([2]);
//         ans.to_view_mut().mul(a.to_view().slice(slice!(..;2)), 2.);
//
//         assert_eq!(ans.index_item([0]), 2.);
//         assert_eq!(ans.index_item([1]), 6.);
//     }
//
//     #[test]
//     fn scalar_2d() {
//         let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
//         let mut ans = OwnedMatrix2D::<f32>::zeros([2, 3]);
//         ans.to_view_mut().mul(a.to_view(), 2.);
//
//         assert_eq!(ans.index_item([0, 0]), 2.);
//         assert_eq!(ans.index_item([0, 1]), 4.);
//         assert_eq!(ans.index_item([0, 2]), 6.);
//         assert_eq!(ans.index_item([1, 0]), 8.);
//         assert_eq!(ans.index_item([1, 1]), 10.);
//         assert_eq!(ans.index_item([1, 2]), 12.);
//     }
//
//     #[test]
//     fn default_1d_1d() {
//         let a = OwnedMatrix1D::from_vec(vec![1., 2., 3.], [3]);
//         let b = OwnedMatrix1D::from_vec(vec![1., 2., 3.], [3]);
//         let mut ans = OwnedMatrix1D::<f32>::zeros([3]);
//         ans.to_view_mut().mul(a.to_view(), b.to_view());
//
//         assert_eq!(ans.index_item([0]), 1.);
//         assert_eq!(ans.index_item([1]), 4.);
//         assert_eq!(ans.index_item([2]), 9.);
//     }
//
//     #[test]
//     fn sliced_1d_1d() {
//         let a = OwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], [4]);
//         let b = OwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], [4]);
//         let mut ans = OwnedMatrix1D::<f32>::zeros([2]);
//         ans.to_view_mut().mul(
//             a.to_view().slice(slice!(..;2)),
//             b.to_view().slice(slice!(..;2)),
//         );
//
//         assert_eq!(ans.index_item([0]), 1.);
//         assert_eq!(ans.index_item([1]), 9.);
//     }
//
//     #[test]
//     fn default_2d_2d() {
//         let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
//         let b = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
//         let mut ans = OwnedMatrix2D::<f32>::zeros([2, 3]);
//         ans.to_view_mut().mul(a.to_view(), b.to_view());
//
//         assert_eq!(ans.index_item([0, 0]), 1.);
//         assert_eq!(ans.index_item([0, 1]), 4.);
//         assert_eq!(ans.index_item([0, 2]), 9.);
//         assert_eq!(ans.index_item([1, 0]), 16.);
//         assert_eq!(ans.index_item([1, 1]), 25.);
//         assert_eq!(ans.index_item([1, 2]), 36.);
//     }
//
//     #[test]
//     fn sliced_4d_2d() {
//         let mut a_vec = Vec::new();
//         for i in 0..2 * 2 * 2 * 2 {
//             a_vec.push(i as f32);
//         }
//
//         let a = OwnedMatrix4D::from_vec(a_vec, [2, 2, 2, 2]);
//         let b = OwnedMatrix1D::from_vec(vec![1., 2.], [2]);
//
//         let mut ans = OwnedMatrix4D::<f32>::zeros([2, 2, 2, 2]);
//
//         ans.to_view_mut().mul(a.to_view(), b.to_view());
//
//         for i in 0..2 {
//             for j in 0..2 {
//                 for k in 0..2 {
//                     for l in 0..2 {
//                         assert_eq!(
//                             ans.index_item([i, j, k, l]),
//                             a.index_item([i, j, k, l]) * b.index_item([l])
//                         );
//                     }
//                 }
//             }
//         }
//     }
//
//     #[test]
//     fn mul_4d_2d_dyn() {
//         let ones_4d = OwnedMatrixDyn::<f32>::ones([2, 2, 2, 2]);
//         let ones_2d = OwnedMatrixDyn::ones([2, 2]);
//         let mut ans = OwnedMatrixDyn::zeros([2, 2, 2, 2]);
//         ans.to_view_mut().mul(ones_4d.to_view(), ones_2d.to_view());
//     }
//
//     #[test]
//     fn default_0d_0d() {
//         let a = OwnedMatrixDyn::from_vec(vec![10.], &[]);
//         let b = OwnedMatrixDyn::from_vec(vec![20.], &[]);
//         let mut ans = OwnedMatrixDyn::<f32>::zeros(&[]);
//         ans.to_view_mut().mul(a.to_view(), b.to_view());
//         assert_eq!(ans.index_item(&[]), 200.);
//     }
//
//     #[test]
//     fn sin_0d() {
//         let a = OwnedMatrixDyn::from_vec(vec![1.0], &[]);
//         let mut ans = OwnedMatrixDyn::<f32>::zeros(&[]);
//         ans.sin(a.to_view());
//         let ans = ans.index_item(&[]);
//         assert!(ans - 1.0_f32.sin() < 1e-6);
//     }
//
//     #[test]
//     fn sin1d() {
//         let a = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0], &[3]);
//         let mut ans = OwnedMatrixDyn::<f32>::zeros(&[3]);
//         ans.sin(a.to_view());
//         assert!(ans.index_item(&[0]) - 1.0_f32.sin() < 1e-6);
//         assert!(ans.index_item(&[1]) - 2.0_f32.sin() < 1e-6);
//         assert!(ans.index_item(&[2]) - 3.0_f32.sin() < 1e-6);
//     }
//
//     #[test]
//     fn sin_2d() {
//         let a = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
//         let mut ans = OwnedMatrixDyn::<f32>::zeros(&[2, 2]);
//         ans.sin(a.to_view());
//         assert!(ans.index_item(&[0, 0]) - 1.0_f32.sin() < 1e-6);
//         assert!(ans.index_item(&[0, 1]) - 2.0_f32.sin() < 1e-6);
//         assert!(ans.index_item(&[1, 0]) - 3.0_f32.sin() < 1e-6);
//         assert!(ans.index_item(&[1, 1]) - 4.0_f32.sin() < 1e-6);
//     }
//
//     #[test]
//     fn sin_3d() {
//         let a = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
//         let mut ans = OwnedMatrixDyn::<f32>::zeros(&[2, 1, 3]);
//         ans.sin(a.to_view());
//         assert!(ans.index_item(&[0, 0, 0]) - 1.0_f32.sin() < 1e-6);
//         assert!(ans.index_item(&[0, 0, 1]) - 2.0_f32.sin() < 1e-6);
//         assert!(ans.index_item(&[0, 0, 2]) - 3.0_f32.sin() < 1e-6);
//         assert!(ans.index_item(&[1, 0, 0]) - 4.0_f32.sin() < 1e-6);
//         assert!(ans.index_item(&[1, 0, 1]) - 5.0_f32.sin() < 1e-6);
//         assert!(ans.index_item(&[1, 0, 2]) - 6.0_f32.sin() < 1e-6);
//     }
// }

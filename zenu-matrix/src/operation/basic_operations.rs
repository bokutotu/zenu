use crate::{
    device::{cpu::Cpu, DeviceBase},
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{Matrix, Owned, Ref, Repr},
    num::Num,
    with_clousers::{
        array_array, array_array_array, array_array_scalar, scalar_array_with_closure,
    },
};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

#[cfg(feature = "nvidia")]
use zenu_cuda::kernel::{
    array_abs, array_abs_assign, array_acos, array_acos_assign, array_add, array_array_add_assign,
    array_array_div_assign, array_array_mul_assign, array_array_sub_assign, array_asin,
    array_asin_assign, array_atan, array_atan_assign, array_cos, array_cos_assign, array_cosh,
    array_cosh_assign, array_div, array_exp, array_exp_assign, array_log, array_log_assign,
    array_mul, array_pow, array_pow_assign, array_scalar_add, array_scalar_add_assign,
    array_scalar_add_assign_ptr, array_scalar_add_ptr, array_scalar_div, array_scalar_div_assign,
    array_scalar_div_assign_ptr, array_scalar_div_ptr, array_scalar_mul, array_scalar_mul_assign,
    array_scalar_mul_assign_ptr, array_scalar_mul_ptr, array_scalar_sub, array_scalar_sub_assign,
    array_scalar_sub_assign_ptr, array_scalar_sub_ptr, array_sin, array_sin_assign, array_sinh,
    array_sinh_assign, array_sqrt, array_sqrt_assign, array_sub, array_tan, array_tan_assign,
    array_tanh, array_tanh_assign,
};

use super::copy_from::CopyBlas;

macro_rules! impl_basic_op_trait {
    (
        $name:ident,
        $cpu_method:ident,
        $cpu_assign_method:ident,
        $gpu_array:ident,
        $gpu_array_assign:ident,
        $gpu_scalar:ident,
        $gpu_scalar_assign:ident,
        $gpu_scalar_ptr:ident,
        $gpu_scalar_assign_ptr:ident
    ) => {
        pub trait $name: DeviceBase {
            fn array_array<T: Num>(
                to: *mut T,
                lhs: *const T,
                rhs: *const T,
                num_elm: usize,
                to_stride: usize,
                lhs_stride: usize,
                rhs_stride: usize,
            );

            fn array_assign<T: Num>(
                to: *mut T,
                rhs: *const T,
                num_elm: usize,
                to_stride: usize,
                rhs_stride: usize,
            );

            fn scalar<T: Num>(
                to: *mut T,
                lhs: *const T,
                rhs: T,
                num_elm: usize,
                to_stride: usize,
                lhs_stride: usize,
            );

            fn scalar_assign<T: Num>(to: *mut T, rhs: T, num_elm: usize, to_stride: usize);

            fn scalar_ptr<T: Num>(
                to: *mut T,
                rhs: *const T,
                scalar: *const T,
                to_stride: usize,
                rhs_stride: usize,
                num_elm: usize
            );

            fn scalar_assign_ptr<T: Num>(
                to: *mut T,
                scalar: *const T,
                num_elm: usize,
                to_stride: usize
            );
        }

        impl$name for Cpu {
            #[expect(clippy::not_unsafe_ptr_arg_deref)]
            fn array_array<T: Num>(
                to: *mut T,
                lhs: *const T,
                rhs: *const T,
                num_elm: usize,
                to_stride: usize,
                lhs_stride: usize,
                rhs_stride: usize,
            ) {
                for i in 0..num_elm {
                    unsafe {
                        *to.add(i * to_stride) =
                            T::$cpu_method(*lhs.add(i * lhs_stride), *rhs.add(i * rhs_stride));
                    }
                }
            }

            #[expect(clippy::not_unsafe_ptr_arg_deref)]
            fn array_assign<T: Num>(
                to: *mut T,
                rhs: *const T,
                num_elm: usize,
                to_stride: usize,
                rhs_stride: usize,
            ) {
                for i in 0..num_elm {
                    unsafe {
                        T::$cpu_assign_method(
                            &mut *to.add(i * to_stride),
                            *rhs.add(i * rhs_stride),
                        );
                    }
                }
            }

            #[expect(clippy::not_unsafe_ptr_arg_deref)]
            fn scalar<T: Num>(
                to: *mut T,
                lhs: *const T,
                rhs: T,
                num_elm: usize,
                to_stride: usize,
                lhs_stride: usize,
            ) {
                for i in 0..num_elm {
                    unsafe {
                        *to.add(i * to_stride) = T::$cpu_method(*lhs.add(i * lhs_stride), rhs);
                    }
                }
            }

            #[expect(clippy::not_unsafe_ptr_arg_deref)]
            fn scalar_assign<T: Num>(to: *mut T, rhs: T, num_elm: usize, to_stride: usize) {
                for i in 0..num_elm {
                    unsafe {
                        T::$cpu_assign_method(&mut *to.add(i * to_stride), rhs);
                    }
                }
            }

            #[expect(clippy::not_unsafe_ptr_arg_deref)]
            fn scalar_ptr<T: Num>(
                to: *mut T,
                lhs: *const T,
                scalar: *const T,
                to_stride: usize,
                lhs_stride: usize,
                num_elm: usize
            ) {
                for i in 0..num_elm {
                    unsafe {
                        *to.add(i * to_stride) = T::$cpu_method(*lhs.add(i * lhs_stride), *scalar);
                    }
                }
            }

            #[expect(clippy::not_unsafe_ptr_arg_deref)]
            fn scalar_assign_ptr<T: Num>(
                to: *mut T,
                scalar: *const T,
                num_elm: usize,
                to_stride: usize
            ) {
                for i in 0..num_elm {
                    unsafe {
                        T::$cpu_assign_method(&mut *to.add(i * to_stride), *scalar);
                    }
                }
            }
        }

        #[cfg(feature = "nvidia")]
        impl $name for Nvidia {
            fn array_array<T: Num>(
                to: *mut T,
                lhs: *const T,
                rhs: *const T,
                num_elm: usize,
                to_stride: usize,
                lhs_stride: usize,
                rhs_stride: usize,
            ) {
                $gpu_array(to, lhs, rhs, num_elm, to_stride, lhs_stride, rhs_stride);
            }

            fn array_assign<T: Num>(
                to: *mut T,
                rhs: *const T,
                num_elm: usize,
                to_stride: usize,
                rhs_stride: usize,
            ) {
                $gpu_array_assign(to, rhs, num_elm, to_stride, rhs_stride);
            }

            fn scalar<T: Num>(
                to: *mut T,
                lhs: *const T,
                rhs: T,
                num_elm: usize,
                to_stride: usize,
                lhs_stride: usize,
            ) {
                $gpu_scalar(to, lhs, rhs, num_elm, to_stride, lhs_stride);
            }

            fn scalar_assign<T: Num>(to: *mut T, rhs: T, num_elm: usize, to_stride: usize) {
                $gpu_scalar_assign(to, rhs, num_elm, to_stride);
            }

            fn scalar_ptr<T: Num>(to: *mut T, lhs: *const T, scalar: *const T, to_stride: usize, lhs_stride: usize, num_elm: usize) {
                $gpu_scalar_ptr(to, lhs, scalar, num_elm, to_stride, lhs_stride);
            }

            fn scalar_assign_ptr<T: Num>(to: *mut T, scalar: *const T, num_elm: usize, to_stride: usize) {
                $gpu_scalar_assign_ptr(to, scalar, num_elm, to_stride);
            }
        }
    };
}
impl_basic_op_trait!(
    AddOps,
    add,
    add_assign,
    array_add,
    array_array_add_assign,
    array_scalar_add,
    array_scalar_add_assign,
    array_scalar_add_ptr,
    array_scalar_add_assign_ptr
);
impl_basic_op_trait!(
    SubOps,
    sub,
    sub_assign,
    array_sub,
    array_array_sub_assign,
    array_scalar_sub,
    array_scalar_sub_assign,
    array_scalar_sub_ptr,
    array_scalar_sub_assign_ptr
);
impl_basic_op_trait!(
    MulOps,
    mul,
    mul_assign,
    array_mul,
    array_array_mul_assign,
    array_scalar_mul,
    array_scalar_mul_assign,
    array_scalar_mul_ptr,
    array_scalar_mul_assign_ptr
);
impl_basic_op_trait!(
    DivOps,
    div,
    div_assign,
    array_div,
    array_array_div_assign,
    array_scalar_div,
    array_scalar_div_assign,
    array_scalar_div_ptr,
    array_scalar_div_assign_ptr
);

macro_rules! impl_basic_ops {
    (
        $method:ident,
        $assign_method:ident,
        $scalar_method:ident,
        $scalar_assign_method:ident,
        $device_trait:ident
    ) => {
        impl<T, D> Matrix<Ref<&mut T>, DimDyn, D>
        where
            T: Num,
            D: DeviceBase + $device_trait,
        {
            pub fn $scalar_method<OR: Repr<Item = T>>(
                &mut self,
                other: &Matrix<OR, DimDyn, D>,
                scalar: T,
            ) {
                array_array_scalar(self, &other.to_ref(), scalar, |a, b, c| {
                    let num_elm = a.shape().num_elm();
                    let to_stride = a.stride().into_iter().last().unwrap_or(1);
                    let rhs_stride = b.stride().into_iter().last().unwrap_or(1);
                    D::scalar(
                        a.as_mut_ptr(),
                        b.as_ptr(),
                        c,
                        num_elm,
                        to_stride,
                        rhs_stride,
                    );
                });
            }
            pub fn $scalar_assign_method(&mut self, scalar: T) {
                scalar_array_with_closure(self, scalar, |a, b| {
                    let num_elm = a.shape().num_elm();
                    let stride = a.stride().into_iter().last().unwrap_or(1);
                    D::scalar_assign(a.as_mut_ptr(), b, num_elm, stride);
                });
            }

            pub fn $assign_method(&mut self, other: &Matrix<Ref<&T>, DimDyn, D>) {
                array_array(
                    self,
                    &other,
                    |a, b| {
                        let num_elm = a.shape().num_elm();
                        let to_stride = a.stride().into_iter().last().unwrap_or(1);
                        let rhs_stride = b.stride().into_iter().last().unwrap_or(1);
                        D::array_assign(a.as_mut_ptr(), b.as_ptr(), num_elm, to_stride, rhs_stride);
                    },
                    |a, b| {
                        let num_elm = a.shape().num_elm();
                        let stride = a.stride().into_iter().last().unwrap_or(1);
                        D::scalar_assign_ptr(a.as_mut_ptr(), b, num_elm, stride);
                    },
                );
            }

            pub fn $method<LR: Repr<Item = T>, RR: Repr<Item = T>>(
                &mut self,
                lhs: &Matrix<LR, DimDyn, D>,
                rhs: &Matrix<RR, DimDyn, D>,
            ) {
                array_array_array(
                    self,
                    &lhs.to_ref(),
                    &rhs.to_ref(),
                    |a, b, c| {
                        let num_elm = a.shape().num_elm();
                        let to_stride = a.stride().into_iter().last().unwrap_or(1);
                        let lhs_stride = b.stride().into_iter().last().unwrap_or(1);
                        let rhs_stride = c.stride().into_iter().last().unwrap_or(1);
                        D::array_array(
                            a.as_mut_ptr(),
                            b.as_ptr(),
                            c.as_ptr(),
                            num_elm,
                            to_stride,
                            lhs_stride,
                            rhs_stride,
                        );
                    },
                    |a, b, c| {
                        let num_elm = a.shape().num_elm();
                        let to_stride = a.stride().into_iter().last().unwrap_or(1);
                        let lhs_stride = b.stride().into_iter().last().unwrap_or(1);
                        D::scalar_ptr(
                            a.as_mut_ptr(),
                            b.as_ptr(),
                            c,
                            to_stride,
                            lhs_stride,
                            num_elm,
                        );
                    },
                );
            }
        }
    };
}
impl_basic_ops!(add_array, add_assign, add_scalar, add_scalar_assign, AddOps);
impl_basic_ops!(sub_array, sub_assign, sub_scalar, sub_scalar_assign, SubOps);
impl_basic_ops!(mul_array, mul_assign, mul_scalar, mul_scalar_assign, MulOps);
impl_basic_ops!(div_array, div_assign, div_scalar, div_scalar_assign, DivOps);

macro_rules! impl_basic_ops_no_inputs {
    ($name:ident, $cpu_method:ident, $gpu_method:ident, $gpu_assign_method:ident) => {
        pub trait $name: DeviceBase {
            fn array<T: Num>(
                to: *mut T,
                other: *const T,
                num_elm: usize,
                to_stride: usize,
                other_stride: usize,
            );

            fn array_assign<T: Num>(to: *mut T, num_elm: usize, to_stride: usize);
        }

        impl $name for Cpu {
            #[expect(clippy::not_unsafe_ptr_arg_deref)]
            fn array<T: Num>(
                to: *mut T,
                other: *const T,
                num_elm: usize,
                to_stride: usize,
                other_stride: usize,
            ) {
                for i in 0..num_elm {
                    unsafe {
                        *to.add(i * to_stride) = T::$cpu_method(*other.add(i * other_stride));
                    }
                }
            }

            #[expect(clippy::not_unsafe_ptr_arg_deref)]
            fn array_assign<T: Num>(to: *mut T, num_elm: usize, to_stride: usize) {
                for i in 0..num_elm {
                    unsafe {
                        *to.add(i * to_stride) = T::$cpu_method(*to.add(i * to_stride));
                    }
                }
            }
        }

        #[cfg(feature = "nvidia")]
        impl $name for Nvidia {
            fn array<T: Num>(
                to: *mut T,
                other: *const T,
                num_elm: usize,
                to_stride: usize,
                other_stride: usize,
            ) {
                $gpu_method(to, other, num_elm, to_stride, other_stride);
            }

            fn array_assign<T: Num>(to: *mut T, num_elm: usize, to_stride: usize) {
                $gpu_assign_method(to, num_elm, to_stride);
            }
        }
    };
}
impl_basic_ops_no_inputs!(SinOps, sin, array_sin, array_sin_assign);
impl_basic_ops_no_inputs!(CosOps, cos, array_cos, array_cos_assign);
impl_basic_ops_no_inputs!(TanOps, tan, array_tan, array_tan_assign);
impl_basic_ops_no_inputs!(AsinOps, asin, array_asin, array_asin_assign);
impl_basic_ops_no_inputs!(AcosOps, acos, array_acos, array_acos_assign);
impl_basic_ops_no_inputs!(AtanOps, atan, array_atan, array_atan_assign);
impl_basic_ops_no_inputs!(SinhOps, sinh, array_sinh, array_sinh_assign);
impl_basic_ops_no_inputs!(CoshOps, cosh, array_cosh, array_cosh_assign);
impl_basic_ops_no_inputs!(TanhOps, tanh, array_tanh, array_tanh_assign);
impl_basic_ops_no_inputs!(AbsOps, abs, array_abs, array_abs_assign);
impl_basic_ops_no_inputs!(SqrtOps, sqrt, array_sqrt, array_sqrt_assign);
impl_basic_ops_no_inputs!(ExpOps, exp, array_exp, array_exp_assign);
impl_basic_ops_no_inputs!(LogOps, ln, array_log, array_log_assign);

pub trait PowOws: DeviceBase {
    fn array<T: Num>(
        to: *mut T,
        other: *const T,
        scalar: T,
        num_elm: usize,
        to_stride: usize,
        other_stride: usize,
    );

    fn array_assign<T: Num>(to: *mut T, scalar: T, num_elm: usize, to_stride: usize);
}
impl PowOws for Cpu {
    #[expect(clippy::not_unsafe_ptr_arg_deref)]
    fn array<T: Num>(
        to: *mut T,
        other: *const T,
        scalar: T,
        num_elm: usize,
        to_stride: usize,
        other_stride: usize,
    ) {
        for i in 0..num_elm {
            unsafe {
                *to.add(i * to_stride) = T::powf(*other.add(i * other_stride), scalar);
            }
        }
    }

    #[expect(clippy::not_unsafe_ptr_arg_deref)]
    fn array_assign<T: Num>(to: *mut T, scalar: T, num_elm: usize, to_stride: usize) {
        for i in 0..num_elm {
            unsafe {
                *to.add(i * to_stride) = T::powf(*to.add(i * to_stride), scalar);
            }
        }
    }
}

#[cfg(feature = "nvidia")]
impl PowOws for Nvidia {
    fn array<T: Num>(
        to: *mut T,
        other: *const T,
        scalar: T,
        num_elm: usize,
        to_stride: usize,
        other_stride: usize,
    ) {
        array_pow(other, num_elm, other_stride, scalar, to, to_stride);
    }

    fn array_assign<T: Num>(to: *mut T, scalar: T, num_elm: usize, to_stride: usize) {
        array_pow_assign(to, num_elm, to_stride, scalar);
    }
}

macro_rules! impl_basic_ops_no_inputs {
    ($trait_name:ident, $output:ident, $method:ident, $assign:ident) => {
        impl<T: Num, S: DimTrait, D: DeviceBase + $trait_name> Matrix<Ref<&mut T>, S, D> {
            pub fn $method<R: Repr<Item = T>, SO: DimTrait>(&self, other: &Matrix<R, SO, D>) {
                if self.shape().slice() != other.shape().slice() {
                    panic!("Matrix shape mismatch");
                }
                if self.shape().is_empty() {
                    D::array(self.as_mut_ptr(), other.as_ptr(), 1, 1, 1);
                } else if self.shape().len() == 1 {
                    D::array(
                        self.as_mut_ptr(),
                        other.as_ptr(),
                        self.shape().num_elm(),
                        self.stride()[0],
                        other.stride()[0],
                    );
                } else {
                    let num_iter = self.shape()[0];
                    for idx in 0..num_iter {
                        let s = self.index_axis_mut_dyn(Index0D::new(idx));
                        let o = other.index_axis_dyn(Index0D::new(idx));
                        s.$method(&o);
                    }
                }
            }

            pub fn $assign(&mut self) {
                if self.shape().is_empty() {
                    D::array_assign(self.as_mut_ptr(), 1, 1);
                } else if self.shape().len() == 1 {
                    D::array_assign(self.as_mut_ptr(), self.shape().num_elm(), self.stride()[0]);
                } else {
                    let num_iter = self.shape()[0];
                    for idx in 0..num_iter {
                        let mut s = self.index_axis_mut_dyn(Index0D::new(idx));
                        s.$assign();
                    }
                }
            }
        }

        impl<T: Num, R: Repr<Item = T>, S: DimTrait, D: DeviceBase + $trait_name> Matrix<R, S, D> {
            pub fn $output(&self) -> Matrix<Owned<T>, S, D> {
                let mut ans = Matrix::alloc(self.shape().clone());
                ans.to_ref_mut().$method(self);
                ans
            }
        }
    };
}
impl_basic_ops_no_inputs!(SinOps, sin, sin_array, sin_assign);
impl_basic_ops_no_inputs!(CosOps, cos, cos_array, cos_assign);
impl_basic_ops_no_inputs!(TanOps, tan, tan_array, tan_assign);
impl_basic_ops_no_inputs!(AsinOps, asin, asin_array, asin_assign);
impl_basic_ops_no_inputs!(AcosOps, acos, acos_array, acos_assign);
impl_basic_ops_no_inputs!(AtanOps, atan, atan_array, atan_assign);
impl_basic_ops_no_inputs!(SinhOps, sinh, sinh_array, sinh_assign);
impl_basic_ops_no_inputs!(CoshOps, cosh, cosh_array, cosh_assign);
impl_basic_ops_no_inputs!(TanhOps, tanh, tanh_array, tanh_assign);
impl_basic_ops_no_inputs!(AbsOps, abs, abs_array, abs_assign);
impl_basic_ops_no_inputs!(SqrtOps, sqrt, sqrt_array, sqrt_assign);
impl_basic_ops_no_inputs!(ExpOps, exp, exp_array, exp_assign);
impl_basic_ops_no_inputs!(LogOps, log, log_array, log_assign);

impl<R: Repr, S: DimTrait, D: DeviceBase + PowOws + CopyBlas> Matrix<R, S, D> {
    pub fn powf_array(&self, scalar: R::Item) -> Matrix<Owned<R::Item>, S, D> {
        let mut powf = Matrix::alloc(self.shape());
        powf.to_ref_mut().powf(self, scalar);
        powf
    }
}

impl<T: Num, S: DimTrait, D: DeviceBase + PowOws + CopyBlas> Matrix<Ref<&mut T>, S, D> {
    #[expect(clippy::missing_panics_doc)]
    pub fn powf<R: Repr<Item = T>, SO: DimTrait>(&self, other: &Matrix<R, SO, D>, scalar: T) {
        assert!(
            self.shape().slice() == other.shape().slice(),
            "Matrix shape mismatch"
        );

        if self.shape().is_empty() {
            D::array(self.as_mut_ptr(), other.as_ptr(), scalar, 1, 1, 1);
        } else if self.shape().len() == 1 {
            D::array(
                self.as_mut_ptr(),
                other.as_ptr(),
                scalar,
                self.shape().num_elm(),
                self.stride()[0],
                other.stride()[0],
            );
        } else {
            let num_iter = self.shape()[0];
            for idx in 0..num_iter {
                let s = self.index_axis_mut_dyn(Index0D::new(idx));
                let o = other.index_axis_dyn(Index0D::new(idx));
                s.powf(&o, scalar);
            }
        }
    }

    pub fn powf_assign(&self, scalar: T) {
        if self.shape().is_empty() {
            D::array_assign(self.as_mut_ptr(), scalar, 1, 1);
        } else if self.shape().len() == 1 {
            D::array_assign(
                self.as_mut_ptr(),
                scalar,
                self.shape().num_elm(),
                self.stride()[0],
            );
        } else {
            let num_iter = self.shape()[0];
            for idx in 0..num_iter {
                let s = self.index_axis_mut_dyn(Index0D::new(idx));
                s.powf_assign(scalar);
            }
        }
    }
}

#[cfg(test)]
mod basic_ops {
    #![expect(clippy::float_cmp, clippy::cast_precision_loss)]
    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
        slice_dynamic,
    };

    // 必要なテスト群
    // default stride
    // sliced
    // transposed

    fn scalar_add_1d<D: Device>() {
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1., 2., 3.], [3]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([3]);
        ans.to_ref_mut().add_scalar(&a, 1.);
        assert_eq!(ans.index_item([0]), 2.);
        assert_eq!(ans.index_item([1]), 3.);
        assert_eq!(ans.index_item([2]), 4.);
    }
    #[test]
    fn scalar_add_1d_cpu() {
        scalar_add_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn scalar_add_1d_gpu() {
        scalar_add_1d::<crate::device::nvidia::Nvidia>();
    }

    fn scalar_add_2d<D: Device>() {
        let a: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 3]);
        ans.to_ref_mut().add_scalar(&a, 1.);
        assert_eq!(ans.index_item([0, 0]), 2.);
        assert_eq!(ans.index_item([0, 1]), 3.);
        assert_eq!(ans.index_item([0, 2]), 4.);
        assert_eq!(ans.index_item([1, 0]), 5.);
        assert_eq!(ans.index_item([1, 1]), 6.);
        assert_eq!(ans.index_item([1, 2]), 7.);
    }
    #[test]
    fn scalar_add_2d_cpu() {
        scalar_add_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn scalar_add_2d_gpu() {
        scalar_add_2d::<crate::device::nvidia::Nvidia>();
    }

    fn sliced_3d<D: Device>() {
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [2, 2, 4],
        );
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([1, 2, 2]);
        let sliced = a.slice(slice_dynamic!(1.., .., ..;2));
        ans.to_ref_mut().add_scalar(&sliced, 1.);
        assert_eq!(ans.index_item([0, 0, 0]), 10.);
        assert_eq!(ans.index_item([0, 0, 1]), 12.);
        assert_eq!(ans.index_item([0, 1, 0]), 14.);
        assert_eq!(ans.index_item([0, 1, 1]), 16.);
    }
    #[test]
    fn sliced_3d_cpu() {
        sliced_3d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sliced_3d_gpu() {
        sliced_3d::<crate::device::nvidia::Nvidia>();
    }

    fn scalar_assign_4d<D: Device>() {
        let mut a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [2, 2, 2, 2],
        );
        a.to_ref_mut().add_scalar_assign(1.);
        assert_eq!(a.index_item([0, 0, 0, 0]), 2.);
        assert_eq!(a.index_item([0, 0, 0, 1]), 3.);
        assert_eq!(a.index_item([0, 0, 1, 0]), 4.);
        assert_eq!(a.index_item([0, 0, 1, 1]), 5.);
        assert_eq!(a.index_item([0, 1, 0, 0]), 6.);
        assert_eq!(a.index_item([0, 1, 0, 1]), 7.);
        assert_eq!(a.index_item([0, 1, 1, 0]), 8.);
        assert_eq!(a.index_item([0, 1, 1, 1]), 9.);
        assert_eq!(a.index_item([1, 0, 0, 0]), 10.);
        assert_eq!(a.index_item([1, 0, 0, 1]), 11.);
        assert_eq!(a.index_item([1, 0, 1, 0]), 12.);
        assert_eq!(a.index_item([1, 0, 1, 1]), 13.);
        assert_eq!(a.index_item([1, 1, 0, 0]), 14.);
        assert_eq!(a.index_item([1, 1, 0, 1]), 15.);
        assert_eq!(a.index_item([1, 1, 1, 0]), 16.);
        assert_eq!(a.index_item([1, 1, 1, 1]), 17.);
    }
    #[test]
    fn scalar_assign_4d_cpu() {
        scalar_assign_4d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn scalar_assign_4d_gpu() {
        scalar_assign_4d::<crate::device::nvidia::Nvidia>();
    }

    fn sliced_3d_assign<D: Device>() {
        let mut a = Vec::new();
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    a.push((i * 100 + j * 10 + k) as f32);
                }
            }
        }
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [3, 4, 5]);
        // shape [1, 2, 3]
        let a = a.slice(slice_dynamic!(2, 1..3, ..;2));
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 3]);
        ans.to_ref_mut().add_scalar(&a, 1.);
        assert_eq!(ans.index_item([0, 0]), 211.);
        assert_eq!(ans.index_item([0, 1]), 213.);
        assert_eq!(ans.index_item([0, 2]), 215.);
        assert_eq!(ans.index_item([1, 0]), 221.);
        assert_eq!(ans.index_item([1, 1]), 223.);
        assert_eq!(ans.index_item([1, 2]), 225.);
    }
    #[test]
    fn sliced_3d_assign_cpu() {
        sliced_3d_assign::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sliced_3d_assign_gpu() {
        sliced_3d_assign::<crate::device::nvidia::Nvidia>();
    }

    fn matrix_add_4d<D: Device>() {
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [2, 2, 2, 2],
        );
        let b: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![
                16., 15., 14., 13., 12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
            ],
            [2, 2, 2, 2],
        );

        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2, 2, 2]);
        ans.to_ref_mut().add_array(&a, &b);
        assert_eq!(ans.index_item([0, 0, 0, 0]), 17.);
        assert_eq!(ans.index_item([0, 0, 0, 1]), 17.);
        assert_eq!(ans.index_item([0, 0, 1, 0]), 17.);
        assert_eq!(ans.index_item([0, 0, 1, 1]), 17.);
        assert_eq!(ans.index_item([0, 1, 0, 0]), 17.);
        assert_eq!(ans.index_item([0, 1, 0, 1]), 17.);
        assert_eq!(ans.index_item([0, 1, 1, 0]), 17.);
        assert_eq!(ans.index_item([0, 1, 1, 1]), 17.);
        assert_eq!(ans.index_item([1, 0, 0, 0]), 17.);
        assert_eq!(ans.index_item([1, 0, 0, 1]), 17.);
        assert_eq!(ans.index_item([1, 0, 1, 0]), 17.);
        assert_eq!(ans.index_item([1, 0, 1, 1]), 17.);
        assert_eq!(ans.index_item([1, 1, 0, 0]), 17.);
        assert_eq!(ans.index_item([1, 1, 0, 1]), 17.);
        assert_eq!(ans.index_item([1, 1, 1, 0]), 17.);
        assert_eq!(ans.index_item([1, 1, 1, 1]), 17.);
    }
    #[test]
    fn matrix_add_4d_cpu() {
        matrix_add_4d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn matrix_add_4d_gpu() {
        matrix_add_4d::<crate::device::nvidia::Nvidia>();
    }

    fn matrix_add_sliced<D: Device>() {
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );
        let b: Matrix<_, DimDyn, _> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );

        let a = a.slice(slice_dynamic!(1..;2, ..;2));
        let b = b.slice(slice_dynamic!(..;2, 1..;2));
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2]);
        ans.to_ref_mut().add_array(&a, &b);
        assert_eq!(ans.index_item([0, 0]), 7.);
        assert_eq!(ans.index_item([0, 1]), 11.);
        assert_eq!(ans.index_item([1, 0]), 23.);
        assert_eq!(ans.index_item([1, 1]), 27.);
    }
    #[test]
    fn matrix_add_sliced_cpu() {
        matrix_add_sliced::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn matrix_add_sliced_gpu() {
        matrix_add_sliced::<crate::device::nvidia::Nvidia>();
    }

    fn transposed<D: Device>() {
        let mut a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );
        let b: Matrix<_, DimDyn, _> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );

        a.transpose();
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([4, 4]);
        ans.to_ref_mut().add_array(&a, &b);
        assert_eq!(ans.index_item([0, 0]), 2.);
        assert_eq!(ans.index_item([0, 1]), 7.);
        assert_eq!(ans.index_item([0, 2]), 12.);
        assert_eq!(ans.index_item([0, 3]), 17.);
        assert_eq!(ans.index_item([1, 0]), 7.);
        assert_eq!(ans.index_item([1, 1]), 12.);
        assert_eq!(ans.index_item([1, 2]), 17.);
        assert_eq!(ans.index_item([1, 3]), 22.);
        assert_eq!(ans.index_item([2, 0]), 12.);
        assert_eq!(ans.index_item([2, 1]), 17.);
        assert_eq!(ans.index_item([2, 2]), 22.);
        assert_eq!(ans.index_item([2, 3]), 27.);
        assert_eq!(ans.index_item([3, 0]), 17.);
        assert_eq!(ans.index_item([3, 1]), 22.);
        assert_eq!(ans.index_item([3, 2]), 27.);
        assert_eq!(ans.index_item([3, 3]), 32.);
    }
    #[test]
    fn transposed_cpu() {
        transposed::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn transposed_gpu() {
        transposed::<crate::device::nvidia::Nvidia>();
    }

    fn broadcast_add<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let b: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1., 1.], [1, 1, 2]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2, 2]);
        ans.to_ref_mut().add_array(&a, &b);
        assert_eq!(ans.index_item([0, 0, 0]), 2.);
        assert_eq!(ans.index_item([0, 0, 1]), 3.);
        assert_eq!(ans.index_item([0, 1, 0]), 4.);
        assert_eq!(ans.index_item([0, 1, 1]), 5.);
        assert_eq!(ans.index_item([1, 0, 0]), 6.);
        assert_eq!(ans.index_item([1, 0, 1]), 7.);
        assert_eq!(ans.index_item([1, 1, 0]), 8.);
        assert_eq!(ans.index_item([1, 1, 1]), 9.);
    }
    #[test]
    fn broadcast_add_cpu() {
        broadcast_add::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn broadcast_add_gpu() {
        broadcast_add::<crate::device::nvidia::Nvidia>();
    }

    fn add_2d_1d<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let b: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1., 1.], [2]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2, 2]);
        ans.to_ref_mut().add_array(&a, &b);
        assert_eq!(ans.index_item([0, 0, 0]), 2.);
        assert_eq!(ans.index_item([0, 0, 1]), 3.);
        assert_eq!(ans.index_item([0, 1, 0]), 4.);
        assert_eq!(ans.index_item([0, 1, 1]), 5.);
        assert_eq!(ans.index_item([1, 0, 0]), 6.);
        assert_eq!(ans.index_item([1, 0, 1]), 7.);
        assert_eq!(ans.index_item([1, 1, 0]), 8.);
        assert_eq!(ans.index_item([1, 1, 1]), 9.);
    }
    #[test]
    fn add_2d_1d_cpu() {
        add_2d_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn add_2d_1d_gpu() {
        add_2d_1d::<crate::device::nvidia::Nvidia>();
    }

    fn add_2d_0d<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let b: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1.], []);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2, 2]);
        ans.to_ref_mut().add_array(&a, &b);
        assert_eq!(ans.index_item([0, 0, 0]), 2.);
        assert_eq!(ans.index_item([0, 0, 1]), 3.);
        assert_eq!(ans.index_item([0, 1, 0]), 4.);
        assert_eq!(ans.index_item([0, 1, 1]), 5.);
        assert_eq!(ans.index_item([1, 0, 0]), 6.);
        assert_eq!(ans.index_item([1, 0, 1]), 7.);
        assert_eq!(ans.index_item([1, 1, 0]), 8.);
        assert_eq!(ans.index_item([1, 1, 1]), 9.);
    }
    #[test]
    fn add_2d_0d_cpu() {
        add_2d_0d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn add_2d_0d_gpu() {
        add_2d_0d::<crate::device::nvidia::Nvidia>();
    }

    fn broad_cast_4x1x1x1_4x3x3x3<D: Device>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 4.], [4, 1, 1, 1]);
        let b = Matrix::<Owned<f32>, DimDyn, D>::zeros([4, 2, 3, 3]);
        let mut ans = Matrix::<Owned<f32>, DimDyn, D>::zeros([4, 2, 3, 3]);
        ans.to_ref_mut().add_array(&a, &b);
        let one = vec![1; 2 * 3 * 3];
        let two = vec![2; 2 * 3 * 3];
        let three = vec![3; 2 * 3 * 3];
        let four = vec![4; 2 * 3 * 3];
        let mut result = Vec::new();
        result.extend_from_slice(&one);
        result.extend_from_slice(&two);
        result.extend_from_slice(&three);
        result.extend_from_slice(&four);
        let result = result.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let result = Matrix::<Owned<f32>, DimDyn, D>::from_vec(result, [4, 2, 3, 3]);
        let diff = ans - result;
        let diff = diff.asum();
        assert!(diff == 0.0);
    }
    #[test]
    fn broad_cast_4x1x1x1_4x3x3x3_cpu() {
        broad_cast_4x1x1x1_4x3x3x3::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn broad_cast_4x1x1x1_4x3x3x3_gpu() {
        broad_cast_4x1x1x1_4x3x3x3::<crate::device::nvidia::Nvidia>();
    }

    fn broadcast_add_1x4x1x1_3x4x5x5<D: Device>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 4.], [1, 4, 1, 1]);
        let b = Matrix::<Owned<f32>, DimDyn, D>::zeros([3, 4, 5, 5]);
        let mut ans = Matrix::<Owned<f32>, DimDyn, D>::zeros([3, 4, 5, 5]);
        ans.to_ref_mut().add_array(&a, &b);
        let one = vec![1; 3 * 5 * 5];
        let two = vec![2; 3 * 5 * 5];
        let three = vec![3; 3 * 5 * 5];
        let four = vec![4; 3 * 5 * 5];
        let mut result = Vec::new();
        result.extend_from_slice(&one);
        result.extend_from_slice(&two);
        result.extend_from_slice(&three);
        result.extend_from_slice(&four);
        let result = result.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let mut result = Matrix::<Owned<f32>, DimDyn, D>::from_vec(result, [4, 3, 5, 5]);
        result.transpose_swap_index(0, 1);
        let diff = (ans - result).asum();
        assert!(diff == 0.0);
    }
    #[test]
    fn broadcast_add_1x4x1x1_3x4x5x5_cpu() {
        broadcast_add_1x4x1x1_3x4x5x5::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn broadcast_add_1x4x1x1_3x4x5x5_gpu() {
        broadcast_add_1x4x1x1_3x4x5x5::<crate::device::nvidia::Nvidia>();
    }

    fn sub_3d_scalar<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2, 2]);
        ans.to_ref_mut().sub_scalar(&a, 1.);

        assert_eq!(ans.index_item([0, 0, 0]), 0.);
        assert_eq!(ans.index_item([0, 0, 1]), 1.);
        assert_eq!(ans.index_item([0, 1, 0]), 2.);
        assert_eq!(ans.index_item([0, 1, 1]), 3.);
        assert_eq!(ans.index_item([1, 0, 0]), 4.);
        assert_eq!(ans.index_item([1, 0, 1]), 5.);
        assert_eq!(ans.index_item([1, 1, 0]), 6.);
        assert_eq!(ans.index_item([1, 1, 1]), 7.);
    }
    #[test]
    fn sub_3d_scalar_cpu() {
        sub_3d_scalar::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sub_3d_scalar_gpu() {
        sub_3d_scalar::<crate::device::nvidia::Nvidia>();
    }

    fn sub_3d_scalar_assign<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let mut a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        a.to_ref_mut().sub_scalar_assign(1.);

        assert_eq!(a.index_item([0, 0, 0]), 0.);
        assert_eq!(a.index_item([0, 0, 1]), 1.);
        assert_eq!(a.index_item([0, 1, 0]), 2.);
        assert_eq!(a.index_item([0, 1, 1]), 3.);
        assert_eq!(a.index_item([1, 0, 0]), 4.);
        assert_eq!(a.index_item([1, 0, 1]), 5.);
        assert_eq!(a.index_item([1, 1, 0]), 6.);
        assert_eq!(a.index_item([1, 1, 1]), 7.);
    }
    #[test]
    fn sub_3d_scalar_assign_cpu() {
        sub_3d_scalar_assign::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sub_3d_scalar_assign_gpu() {
        sub_3d_scalar_assign::<crate::device::nvidia::Nvidia>();
    }

    fn sub_3d_array<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let b = vec![1., 1., 1., 1., 1., 1., 1., 1.];
        let b: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(b, [2, 2, 2]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2, 2]);
        ans.to_ref_mut().sub_array(&a, &b);

        assert_eq!(ans.index_item([0, 0, 0]), 0.);
        assert_eq!(ans.index_item([0, 0, 1]), 1.);
        assert_eq!(ans.index_item([0, 1, 0]), 2.);
        assert_eq!(ans.index_item([0, 1, 1]), 3.);
        assert_eq!(ans.index_item([1, 0, 0]), 4.);
        assert_eq!(ans.index_item([1, 0, 1]), 5.);
        assert_eq!(ans.index_item([1, 1, 0]), 6.);
        assert_eq!(ans.index_item([1, 1, 1]), 7.);
    }
    #[test]
    fn sub_3d_array_cpu() {
        sub_3d_array::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sub_3d_array_gpu() {
        sub_3d_array::<crate::device::nvidia::Nvidia>();
    }

    fn sub_assign_array_3d<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let mut a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let b = vec![1., 1., 1., 1., 1., 1., 1., 1.];
        let b: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(b, [2, 2, 2]);
        a.to_ref_mut().sub_assign(&b.to_ref());

        assert_eq!(a.index_item([0, 0, 0]), 0.);
        assert_eq!(a.index_item([0, 0, 1]), 1.);
        assert_eq!(a.index_item([0, 1, 0]), 2.);
        assert_eq!(a.index_item([0, 1, 1]), 3.);
        assert_eq!(a.index_item([1, 0, 0]), 4.);
        assert_eq!(a.index_item([1, 0, 1]), 5.);
        assert_eq!(a.index_item([1, 1, 0]), 6.);
        assert_eq!(a.index_item([1, 1, 1]), 7.);
    }
    #[test]
    fn sub_assign_array_3d_cpu() {
        sub_assign_array_3d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sub_assign_array_3d_gpu() {
        sub_assign_array_3d::<crate::device::nvidia::Nvidia>();
    }

    fn mul_scalar<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2, 2]);
        ans.to_ref_mut().mul_scalar(&a, 2.);

        assert_eq!(ans.index_item([0, 0, 0]), 2.);
        assert_eq!(ans.index_item([0, 0, 1]), 4.);
        assert_eq!(ans.index_item([0, 1, 0]), 6.);
        assert_eq!(ans.index_item([0, 1, 1]), 8.);
        assert_eq!(ans.index_item([1, 0, 0]), 10.);
        assert_eq!(ans.index_item([1, 0, 1]), 12.);
        assert_eq!(ans.index_item([1, 1, 0]), 14.);
        assert_eq!(ans.index_item([1, 1, 1]), 16.);
    }
    #[test]
    fn mul_scalar_cpu() {
        mul_scalar::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn mul_scalar_gpu() {
        mul_scalar::<crate::device::nvidia::Nvidia>();
    }

    fn mul_scalar_assign<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let mut a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        a.to_ref_mut().mul_scalar_assign(2.);

        assert_eq!(a.index_item([0, 0, 0]), 2.);
        assert_eq!(a.index_item([0, 0, 1]), 4.);
        assert_eq!(a.index_item([0, 1, 0]), 6.);
        assert_eq!(a.index_item([0, 1, 1]), 8.);
        assert_eq!(a.index_item([1, 0, 0]), 10.);
        assert_eq!(a.index_item([1, 0, 1]), 12.);
        assert_eq!(a.index_item([1, 1, 0]), 14.);
        assert_eq!(a.index_item([1, 1, 1]), 16.);
    }
    #[test]
    fn mul_scalar_assign_cpu() {
        mul_scalar_assign::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn mul_scalar_assign_gpu() {
        mul_scalar_assign::<crate::device::nvidia::Nvidia>();
    }

    fn mul_array<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let b = vec![8., 7., 6., 5., 4., 3., 2., 1.];
        let b: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(b, [2, 2, 2]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2, 2]);
        ans.to_ref_mut().mul_array(&a, &b);

        assert_eq!(ans.index_item([0, 0, 0]), 8.);
        assert_eq!(ans.index_item([0, 0, 1]), 14.);
        assert_eq!(ans.index_item([0, 1, 0]), 18.);
        assert_eq!(ans.index_item([0, 1, 1]), 20.);
        assert_eq!(ans.index_item([1, 0, 0]), 20.);
        assert_eq!(ans.index_item([1, 0, 1]), 18.);
        assert_eq!(ans.index_item([1, 1, 0]), 14.);
        assert_eq!(ans.index_item([1, 1, 1]), 8.);
    }
    #[test]
    fn mul_array_cpu() {
        mul_array::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn mul_array_gpu() {
        mul_array::<crate::device::nvidia::Nvidia>();
    }

    fn mul_assign_array<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let mut a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let b = vec![8., 7., 6., 5., 4., 3., 2., 1.];
        let b: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(b, [2, 2, 2]);
        a.to_ref_mut().mul_assign(&b.to_ref());

        assert_eq!(a.index_item([0, 0, 0]), 8.);
        assert_eq!(a.index_item([0, 0, 1]), 14.);
        assert_eq!(a.index_item([0, 1, 0]), 18.);
        assert_eq!(a.index_item([0, 1, 1]), 20.);
        assert_eq!(a.index_item([1, 0, 0]), 20.);
        assert_eq!(a.index_item([1, 0, 1]), 18.);
        assert_eq!(a.index_item([1, 1, 0]), 14.);
        assert_eq!(a.index_item([1, 1, 1]), 8.);
    }
    #[test]
    fn mul_assign_array_cpu() {
        mul_assign_array::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn mul_assign_array_gpu() {
        mul_assign_array::<crate::device::nvidia::Nvidia>();
    }

    fn div_scalar<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2, 2]);
        ans.to_ref_mut().div_scalar(&a, 2.);

        assert_eq!(ans.index_item([0, 0, 0]), 0.5);
        assert_eq!(ans.index_item([0, 0, 1]), 1.);
        assert_eq!(ans.index_item([0, 1, 0]), 1.5);
        assert_eq!(ans.index_item([0, 1, 1]), 2.);
        assert_eq!(ans.index_item([1, 0, 0]), 2.5);
        assert_eq!(ans.index_item([1, 0, 1]), 3.);
        assert_eq!(ans.index_item([1, 1, 0]), 3.5);
        assert_eq!(ans.index_item([1, 1, 1]), 4.);
    }
    #[test]
    fn div_scalar_cpu() {
        div_scalar::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn div_scalar_gpu() {
        div_scalar::<crate::device::nvidia::Nvidia>();
    }

    fn div_scalar_assign<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let mut a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        a.to_ref_mut().div_scalar_assign(2.);

        assert_eq!(a.index_item([0, 0, 0]), 0.5);
        assert_eq!(a.index_item([0, 0, 1]), 1.);
        assert_eq!(a.index_item([0, 1, 0]), 1.5);
        assert_eq!(a.index_item([0, 1, 1]), 2.);
        assert_eq!(a.index_item([1, 0, 0]), 2.5);
        assert_eq!(a.index_item([1, 0, 1]), 3.);
        assert_eq!(a.index_item([1, 1, 0]), 3.5);
        assert_eq!(a.index_item([1, 1, 1]), 4.);
    }
    #[test]
    fn div_scalar_assign_cpu() {
        div_scalar_assign::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn div_scalar_assign_gpu() {
        div_scalar_assign::<crate::device::nvidia::Nvidia>();
    }

    fn div_array<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let b = vec![8., 7., 6., 5., 4., 3., 2., 1.];
        let b: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(b, [2, 2, 2]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2, 2]);
        ans.to_ref_mut().div_array(&a, &b);

        assert_eq!(ans.index_item([0, 0, 0]), 1. / 8.);
        assert_eq!(ans.index_item([0, 0, 1]), 2. / 7.);
        assert_eq!(ans.index_item([0, 1, 0]), 3. / 6.);
        assert_eq!(ans.index_item([0, 1, 1]), 4. / 5.);
        assert_eq!(ans.index_item([1, 0, 0]), 5. / 4.);
        assert_eq!(ans.index_item([1, 0, 1]), 6. / 3.);
        assert_eq!(ans.index_item([1, 1, 0]), 7. / 2.);
        assert_eq!(ans.index_item([1, 1, 1]), 8. / 1.);
    }
    #[test]
    fn div_array_cpu() {
        div_array::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn div_array_gpu() {
        div_array::<crate::device::nvidia::Nvidia>();
    }

    fn div_assign_array<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let mut a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let b = vec![8., 7., 6., 5., 4., 3., 2., 1.];
        let b: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(b, [2, 2, 2]);
        a.to_ref_mut().div_assign(&b.to_ref());

        assert_eq!(a.index_item([0, 0, 0]), 1. / 8.);
        assert_eq!(a.index_item([0, 0, 1]), 2. / 7.);
        assert_eq!(a.index_item([0, 1, 0]), 3. / 6.);
        assert_eq!(a.index_item([0, 1, 1]), 4. / 5.);
        assert_eq!(a.index_item([1, 0, 0]), 5. / 4.);
        assert_eq!(a.index_item([1, 0, 1]), 6. / 3.);
        assert_eq!(a.index_item([1, 1, 0]), 7. / 2.);
        assert_eq!(a.index_item([1, 1, 1]), 8. / 1.);
    }
    #[test]
    fn div_assign_array_cpu() {
        div_assign_array::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn div_assign_array_gpu() {
        div_assign_array::<crate::device::nvidia::Nvidia>();
    }

    fn sin_3d<D: Device>() {
        let a = vec![0., 1., 2., 3., 4., 5., 6., 7.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 2, 2]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([2, 2, 2]);
        ans.to_ref_mut().sin_array(&a);

        // convert this test code to epsilon comparison
        assert!(ans.index_item([0, 0, 0]) - 0. < 1e-6);
        assert!((ans.index_item([0, 0, 1]) - f32::sin(1.)).abs() < 1e-6);
        assert!((ans.index_item([0, 1, 0]) - f32::sin(2.)).abs() < 1e-6);
        assert!((ans.index_item([0, 1, 1]) - f32::sin(3.)).abs() < 1e-6);
        assert!((ans.index_item([1, 0, 0]) - f32::sin(4.)).abs() < 1e-6);
        assert!((ans.index_item([1, 0, 1]) - f32::sin(5.)).abs() < 1e-6);
        assert!((ans.index_item([1, 1, 0]) - f32::sin(6.)).abs() < 1e-6);
        assert!((ans.index_item([1, 1, 1]) - f32::sin(7.)).abs() < 1e-6);
    }
    #[test]
    fn sin_3d_cpu() {
        sin_3d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sin_3d_gpu() {
        sin_3d::<crate::device::nvidia::Nvidia>();
    }

    fn sin_1d_sliced<D: Device>() {
        let a = vec![0., 1., 2., 3., 4., 5., 6., 7.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [8]);
        let a = a.slice(slice_dynamic![1..;2]);
        let mut ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::zeros([4]);
        ans.to_ref_mut().sin_array(&a);

        assert!((ans.index_item([0]) - f32::sin(1.)).abs() < 1e-6);
        assert!((ans.index_item([1]) - f32::sin(3.)).abs() < 1e-6);
        assert!((ans.index_item([2]) - f32::sin(5.)).abs() < 1e-6);
        assert!((ans.index_item([3]) - f32::sin(7.)).abs() < 1e-6);
    }
    #[test]
    fn sin_1d_sliced_cpu() {
        sin_1d_sliced::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sin_1d_sliced_gpu() {
        sin_1d_sliced::<crate::device::nvidia::Nvidia>();
    }

    fn pow_1d<D: Device>() {
        let a = vec![0., 1., 2., 3., 4., 5., 6., 7.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [8]);
        let ans = a.powf_array(2.);
        assert_eq!(ans.index_item([0]), 0.);
        assert_eq!(ans.index_item([1]), 1.);
        assert_eq!(ans.index_item([2]), 4.);
        assert_eq!(ans.index_item([3]), 9.);
        assert_eq!(ans.index_item([4]), 16.);
        assert_eq!(ans.index_item([5]), 25.);
        assert_eq!(ans.index_item([6]), 36.);
        assert_eq!(ans.index_item([7]), 49.);
    }
    #[test]
    fn pow_1d_cpu() {
        pow_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn pow_1d_gpu() {
        pow_1d::<crate::device::nvidia::Nvidia>();
    }

    fn pow_0d<D: Device>() {
        let a = vec![4.];
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, []);
        let ans = a.powf_array(2.);
        assert_eq!(ans.index_item([]), 16.);
    }
    #[test]
    fn pow_0d_cpu() {
        pow_0d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn pow_0d_gpu() {
        pow_0d::<crate::device::nvidia::Nvidia>();
    }

    fn pow_2d_transposed<D: Device>() {
        let a = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let mut a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(a, [2, 4]);
        a.transpose();
        let ans = a.powf_array(2.);
        assert_eq!(ans.index_item([0, 0]), 1.);
        assert_eq!(ans.index_item([0, 1]), 25.);
        assert_eq!(ans.index_item([1, 0]), 4.);
        assert_eq!(ans.index_item([1, 1]), 36.);
        assert_eq!(ans.index_item([2, 0]), 9.);
        assert_eq!(ans.index_item([2, 1]), 49.);
        assert_eq!(ans.index_item([3, 0]), 16.);
        assert_eq!(ans.index_item([3, 1]), 64.);
    }
    #[test]
    fn pow_2d_transposed_cpu() {
        pow_2d_transposed::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn pow_2d_transposed_gpu() {
        pow_2d_transposed::<crate::device::nvidia::Nvidia>();
    }
}

use crate::{
    device::{cpu::Cpu, DeviceBase},
    dim::{larger_shape, smaller_shape, DimDyn, DimTrait},
    index::Index0D,
    matrix::{Matrix, Ref, Repr},
    num::Num,
};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

#[cfg(feature = "nvidia")]
use zenu_cuda::kernel::*;

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

macro_rules! impl_basic_op_trait {
    ($name:ident, $cpu_method:ident, $cpu_assign_method:ident, $gpu_array:ident, $gpu_array_assign:ident, $gpu_scalar:ident, $gpu_scalar_assign:ident) => {
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
        }

        impl$name for Cpu {
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

            fn scalar_assign<T: Num>(to: *mut T, rhs: T, num_elm: usize, to_stride: usize) {
                for i in 0..num_elm {
                    unsafe {
                        T::$cpu_assign_method(&mut *to.add(i * to_stride), rhs);
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
    array_scalar_add_assign
);
impl_basic_op_trait!(
    SubOps,
    sub,
    sub_assign,
    array_sub,
    array_array_sub_assign,
    array_scalar_sub,
    array_scalar_sub_assign
);
impl_basic_op_trait!(
    MulOps,
    mul,
    mul_assign,
    array_mul,
    array_array_mul_assign,
    array_scalar_mul,
    array_scalar_mul_assign
);
impl_basic_op_trait!(
    DivOps,
    div,
    div_assign,
    array_div,
    array_array_div_assign,
    array_scalar_div,
    array_scalar_div_assign
);

/// 1dのMatrixを受け取る(これは入力側でチェック)
/// その配列の中身が1かどうかを確認
/// 1ならtrueを返す
fn is_1d_1(a: &[usize]) -> bool {
    a[0] == 1
}

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
            D: DeviceBase + $device_trait,
        {
            pub fn $scalar_method<RL: Repr<Item=T>, SL: DimTrait>(&mut self, lhs: &Matrix<RL, SL, D>, rhs: T) {
                if self.shape().slice() != lhs.shape().slice() {
                    panic!("Matrix shape mismatch");
                }

                if self.shape().is_empty() {
                    D::scalar(
                        self.as_mut_ptr(),
                        lhs.as_ptr(),
                        rhs,
                        self.shape().num_elm(),
                        // self.stride()[0],
                        // lhs.stride()[0]
                        1,
                        1
                    );
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
                        s.$scalar_method(&lhs, rhs);
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

            pub fn $method<RL, RR, SL, SR>(&mut self, lhs: &Matrix<RL, SL, D>, rhs: &Matrix<RR, SR, D>)
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
                    self.$scalar_method(lhs, rhs.index_item(&[] as &[usize]));
                    return;
                }

                if lhs.shape().is_empty() {
                    self.$scalar_method(rhs, lhs.index_item(&[] as &[usize]));
                    return;
                }

                if self.shape().is_empty() {
                    // let self_slice = self.as_mut_slice();
                    // let lhs_slice = lhs.as_slice();
                    // let rhs_slice = rhs.as_slice();
                    // self_slice[0] = lhs_slice[0].$method(rhs_slice[0]);
                    D::scalar(
                        self.as_mut_ptr(),
                        lhs.as_ptr(),
                        unsafe { *rhs.as_ptr() },
                        // self.shape().num_elm(),
                        1,
                        // self.stride()[0],
                        // lhs.stride()[0],
                        1,
                        1,
                    );
                } else if self.shape().len() == 1 {
                    if is_1d_1(lhs.shape().slice()) {
                        D::scalar(
                            self.as_mut_ptr(),
                            rhs.as_ptr(),
                            lhs.index_item(&[0 as usize] as &[usize]),
                            self.shape().num_elm(),
                            self.stride()[0],
                            rhs.stride()[0]
                        );
                        return;
                    } else if is_1d_1(rhs.shape().slice()) {
                        D::scalar(
                            self.as_mut_ptr(),
                            lhs.as_ptr(),
                            rhs.index_item(&[0 as usize] as &[usize]),
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
                        s.$method(&lhs, &rhs);
                    }
                }
            }

            pub fn $assign_method<RR, SR>(&mut self, rhs: &Matrix<RR, SR, D>)
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
                            rhs.index_item(&[0 as usize] as &[usize])
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
                        s.$assign_method(&rhs);
                    }
                }
            }

        }
    };
}
impl_basic_ops!(add_array, add_assign, add_scalar, add_scalar_assign, AddOps);
impl_basic_ops!(sub_array, sub_assign, sub_scalar, sub_scalar_assign, SubOps);
impl_basic_ops!(mul_array, mul_assign, mul_scalar, mul_scalar_assign, MulOps);
impl_basic_ops!(div_array, div_assign, div_scalar, div_scalar_assign, DivOps);

#[cfg(test)]
mod add {

    use crate::{
        dim::{Dim0, Dim1, Dim2, Dim3},
        matrix::Owned,
        slice,
    };

    use super::*;

    #[test]
    fn add_0d_0d() {
        let a: Matrix<Owned<f32>, Dim0, Cpu> = Matrix::from_vec(vec![1.0], []);
        let b: Matrix<Owned<f32>, Dim0, Cpu> = Matrix::from_vec(vec![1.0], []);
        let mut ans: Matrix<Owned<f32>, Dim0, Cpu> = Matrix::zeros([]);
        ans.to_ref_mut().add_array(&a, &b);
        assert_eq!(ans.index_item([]), 2.0);
    }

    #[test]
    fn add_dyn_dyn() {
        let a: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let ans: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::zeros([3]);

        let a = a.into_dyn_dim();
        let b = b.into_dyn_dim();
        let mut ans = ans.into_dyn_dim();

        ans.to_ref_mut().add_array(&a.to_ref(), &b.to_ref());
    }

    #[test]
    fn add_1d_scalar() {
        let a: Matrix<Owned<f32>, Dim1, Cpu> = Matrix::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let mut ans: Matrix<Owned<f32>, Dim1, Cpu> = Matrix::zeros([3]);
        let b: Matrix<Owned<f32>, Dim0, Cpu> = Matrix::from_vec(vec![2.0], []);
        ans.to_ref_mut().add_array(&a, &b);

        assert_eq!(ans.index_item([0]), 3.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 5.0);
    }

    #[test]
    fn add_1d_scalar_default_stride() {
        let a: Matrix<Owned<f32>, Dim1, Cpu> = Matrix::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let mut ans: Matrix<Owned<f32>, Dim1, Cpu> = Matrix::zeros([3]);
        ans.to_ref_mut().add_scalar(&a, 2.0);

        assert_eq!(ans.index_item([0]), 3.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 5.0);
    }

    #[test]
    fn add_1d_scalar_sliced() {
        let a: Matrix<Owned<f32>, Dim1, Cpu> =
            Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
        let mut ans: Matrix<Owned<f32>, Dim1, Cpu> = Matrix::zeros([3]);
        let sliced = a.slice(slice!(..;2));
        ans.to_ref_mut().add_scalar(&sliced, 1.);
        assert_eq!(ans.index_item([0]), 2.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 6.0);
    }

    #[test]
    fn add_3d_scalar_sliced() {
        let a: Matrix<Owned<f32>, Dim3, Cpu> = Matrix::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
            ],
            [3, 3, 4],
        );

        let mut ans: Matrix<Owned<f32>, Dim3, Cpu> = Matrix::zeros([3, 3, 2]);

        let sliced = a.slice(slice!(.., .., ..;2));

        ans.to_ref_mut().add_scalar(&sliced, 1.);

        assert_eq!(ans.index_item([0, 0, 0]), 2.0);
        assert_eq!(ans.index_item([0, 0, 1]), 4.0);
        assert_eq!(ans.index_item([0, 1, 0]), 6.0);
        assert_eq!(ans.index_item([0, 1, 1]), 8.0);
        assert_eq!(ans.index_item([0, 2, 0]), 10.0);
        assert_eq!(ans.index_item([0, 2, 1]), 12.0);
        assert_eq!(ans.index_item([1, 0, 0]), 14.0);
        assert_eq!(ans.index_item([1, 0, 1]), 16.0);
        assert_eq!(ans.index_item([1, 1, 0]), 18.0);
        assert_eq!(ans.index_item([1, 1, 1]), 20.0);
        assert_eq!(ans.index_item([1, 2, 0]), 22.0);
        assert_eq!(ans.index_item([1, 2, 1]), 24.0);
        assert_eq!(ans.index_item([2, 0, 0]), 26.0);
        assert_eq!(ans.index_item([2, 0, 1]), 28.0);
        assert_eq!(ans.index_item([2, 1, 0]), 30.0);
        assert_eq!(ans.index_item([2, 1, 1]), 32.0);
        assert_eq!(ans.index_item([2, 2, 0]), 34.0);
        assert_eq!(ans.index_item([2, 2, 1]), 36.0);
    }

    #[test]
    fn add_1d_1d_default_stride() {
        let a: Matrix<Owned<f32>, Dim1, Cpu> = Matrix::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b: Matrix<Owned<f32>, Dim1, Cpu> = Matrix::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let mut ans: Matrix<Owned<f32>, Dim1, Cpu> = Matrix::zeros([3]);
        ans.to_ref_mut().add_array(&a, &b);
        assert_eq!(ans.index_item([0]), 2.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 6.0);
    }

    #[test]
    fn add_1d_1d_sliced() {
        let a: Matrix<Owned<f32>, Dim1, Cpu> =
            Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
        let b: Matrix<Owned<f32>, Dim1, Cpu> =
            Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
        let sliced_a = a.slice(slice!(..;2));
        let sliced_b = b.slice(slice!(1..;2));
        let mut ans: Matrix<Owned<f32>, Dim1, Cpu> = Matrix::zeros([3]);
        ans.to_ref_mut().add_array(&sliced_a, &sliced_b);
        assert_eq!(ans.index_item([0]), 3.0);
        assert_eq!(ans.index_item([1]), 7.0);
        assert_eq!(ans.index_item([2]), 11.0);
    }

    #[test]
    fn add_2d_1d_default() {
        let a: Matrix<Owned<f32>, Dim2, Cpu> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );
        let b: Matrix<Owned<f32>, Dim1, Cpu> =
            Matrix::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], [8]);
        let mut ans: Matrix<Owned<f32>, Dim2, Cpu> = Matrix::zeros([2, 2]);
        let sliced_a = a.slice(slice!(..2, ..2));
        let sliced_b = b.slice(slice!(..2));
        ans.to_ref_mut().add_array(&sliced_a, &sliced_b);
        assert_eq!(ans.index_item([0, 0]), 2.0);
        assert_eq!(ans.index_item([0, 1]), 4.0);
        assert_eq!(ans.index_item([1, 0]), 6.0);
        assert_eq!(ans.index_item([1, 1]), 8.0);
    }

    #[test]
    fn add_3d_1d_sliced() {
        let mut v = Vec::new();
        let num_elm = 4 * 4 * 4;
        for i in 0..num_elm {
            v.push(i as f32);
        }
        let a: Matrix<Owned<f32>, Dim3, Cpu> = Matrix::from_vec(v, [4, 4, 4]);
        let b: Matrix<Owned<f32>, Dim1, Cpu> = Matrix::from_vec(vec![1., 2., 3., 4.], [4]);
        let mut ans: Matrix<Owned<f32>, Dim3, Cpu> = Matrix::zeros([2, 2, 2]);
        let sliced_a = a.slice(slice!(..2, 1..;2, ..2));
        let sliced_b = b.slice(slice!(..2));

        ans.to_ref_mut().add_array(&sliced_a, &sliced_b);

        assert_eq!(ans.index_item([0, 0, 0]), 5.);
        assert_eq!(ans.index_item([0, 0, 1]), 7.);
        assert_eq!(ans.index_item([0, 1, 0]), 13.);
        assert_eq!(ans.index_item([0, 1, 1]), 15.);
        assert_eq!(ans.index_item([1, 0, 0]), 21.);
        assert_eq!(ans.index_item([1, 0, 1]), 23.);
        assert_eq!(ans.index_item([1, 1, 0]), 29.);
        assert_eq!(ans.index_item([1, 1, 1]), 31.);
    }

    #[test]
    fn add_2d_2d_default() {
        let a: Matrix<Owned<f32>, Dim2, Cpu> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );
        let b: Matrix<Owned<f32>, Dim2, Cpu> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );
        let mut ans: Matrix<Owned<f32>, Dim2, Cpu> = Matrix::zeros([4, 4]);
        ans.to_ref_mut().add_array(&a, &b);
        assert_eq!(ans.index_item([0, 0]), 2.0);
        assert_eq!(ans.index_item([0, 1]), 4.0);
        assert_eq!(ans.index_item([0, 2]), 6.0);
        assert_eq!(ans.index_item([0, 3]), 8.0);
        assert_eq!(ans.index_item([1, 0]), 10.0);
        assert_eq!(ans.index_item([1, 1]), 12.0);
        assert_eq!(ans.index_item([1, 2]), 14.0);
        assert_eq!(ans.index_item([1, 3]), 16.0);
        assert_eq!(ans.index_item([2, 0]), 18.0);
        assert_eq!(ans.index_item([2, 1]), 20.0);
        assert_eq!(ans.index_item([2, 2]), 22.0);
        assert_eq!(ans.index_item([2, 3]), 24.0);
        assert_eq!(ans.index_item([3, 0]), 26.0);
        assert_eq!(ans.index_item([3, 1]), 28.0);
        assert_eq!(ans.index_item([3, 2]), 30.0);
        assert_eq!(ans.index_item([3, 3]), 32.0);
    }

    #[test]
    fn add_2d_0d() {
        let a: Matrix<Owned<f32>, Dim2, Cpu> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );
        let b: Matrix<Owned<f32>, Dim0, Cpu> = Matrix::from_vec(vec![1.], []);
        let mut ans: Matrix<Owned<f32>, Dim2, Cpu> = Matrix::zeros([4, 4]);
        ans.to_ref_mut().add_array(&a, &b);
        assert_eq!(ans.index_item([0, 0]), 2.0);
        assert_eq!(ans.index_item([0, 1]), 3.0);
        assert_eq!(ans.index_item([0, 2]), 4.0);
        assert_eq!(ans.index_item([0, 3]), 5.0);
        assert_eq!(ans.index_item([1, 0]), 6.0);
        assert_eq!(ans.index_item([1, 1]), 7.0);
        assert_eq!(ans.index_item([1, 2]), 8.0);
        assert_eq!(ans.index_item([1, 3]), 9.0);
        assert_eq!(ans.index_item([2, 0]), 10.0);
        assert_eq!(ans.index_item([2, 1]), 11.0);
        assert_eq!(ans.index_item([2, 2]), 12.0);
        assert_eq!(ans.index_item([2, 3]), 13.0);
        assert_eq!(ans.index_item([3, 0]), 14.0);
        assert_eq!(ans.index_item([3, 1]), 15.0);
        assert_eq!(ans.index_item([3, 2]), 16.0);
        assert_eq!(ans.index_item([3, 3]), 17.0);
    }

    #[test]
    fn add_2d_0d_dyn() {
        let a: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );
        let b: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::from_vec(vec![1.], []);
        let mut ans: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::zeros([4, 4]);
        ans.to_ref_mut().add_array(&a, &b);
        assert_eq!(ans.index_item([0, 0]), 2.0);
        assert_eq!(ans.index_item([0, 1]), 3.0);
        assert_eq!(ans.index_item([0, 2]), 4.0);
        assert_eq!(ans.index_item([0, 3]), 5.0);
        assert_eq!(ans.index_item([1, 0]), 6.0);
        assert_eq!(ans.index_item([1, 1]), 7.0);
        assert_eq!(ans.index_item([1, 2]), 8.0);
        assert_eq!(ans.index_item([1, 3]), 9.0);
        assert_eq!(ans.index_item([2, 0]), 10.0);
        assert_eq!(ans.index_item([2, 1]), 11.0);
        assert_eq!(ans.index_item([2, 2]), 12.0);
        assert_eq!(ans.index_item([2, 3]), 13.0);
        assert_eq!(ans.index_item([3, 0]), 14.0);
        assert_eq!(ans.index_item([3, 1]), 15.0);
        assert_eq!(ans.index_item([3, 2]), 16.0);
        assert_eq!(ans.index_item([3, 3]), 17.0);
    }

    #[test]
    fn add_4d_2d_dyn() {
        let zeros_4d: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::zeros([2, 2, 2, 2]);
        let ones_2d: Matrix<Owned<f32>, DimDyn, Cpu> =
            Matrix::from_vec(vec![1., 1., 1., 1.], [2, 2]);
        let mut ans: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::zeros([2, 2, 2, 2]);
        ans.to_ref_mut().add_array(&zeros_4d, &ones_2d);
        assert_eq!(ans.index_item([0, 0, 0, 0]), 1.0);
        assert_eq!(ans.index_item([0, 0, 0, 1]), 1.0);
        assert_eq!(ans.index_item([0, 0, 1, 0]), 1.0);
        assert_eq!(ans.index_item([0, 0, 1, 1]), 1.0);
        assert_eq!(ans.index_item([0, 1, 0, 0]), 1.0);
        assert_eq!(ans.index_item([0, 1, 0, 1]), 1.0);
        assert_eq!(ans.index_item([0, 1, 1, 0]), 1.0);
        assert_eq!(ans.index_item([0, 1, 1, 1]), 1.0);
        assert_eq!(ans.index_item([1, 0, 0, 0]), 1.0);
        assert_eq!(ans.index_item([1, 0, 0, 1]), 1.0);
        assert_eq!(ans.index_item([1, 0, 1, 0]), 1.0);
        assert_eq!(ans.index_item([1, 0, 1, 1]), 1.0);
        assert_eq!(ans.index_item([1, 1, 0, 0]), 1.0);
        assert_eq!(ans.index_item([1, 1, 0, 1]), 1.0);
        assert_eq!(ans.index_item([1, 1, 1, 0]), 1.0);
        assert_eq!(ans.index_item([1, 1, 1, 1]), 1.0);
    }

    #[test]
    fn broad_cast_4x1x1x1_4x3x3x3() {
        let a: Matrix<Owned<f32>, DimDyn, Cpu> =
            Matrix::from_vec(vec![1., 2., 3., 4.], [4, 1, 1, 1]);
        let b: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::zeros([4, 2, 3, 3]);
        let mut ans: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::zeros([4, 2, 3, 3]);
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
        let result: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::from_vec(result, [4, 2, 3, 3]);
        // assert!((ans.to_ref() - result.to_ref()).asum() == 0.0);
    }
}

#[cfg(test)]
mod sub {
    use crate::{dim::DimDyn, matrix::Owned};

    use super::*;

    #[test]
    fn sub_0d_0d() {
        let a: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::from_vec(vec![1.0], []);
        let b: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::from_vec(vec![1.0], []);
        let mut ans: Matrix<Owned<f32>, DimDyn, Cpu> = Matrix::zeros([]);
        ans.to_ref_mut().sub_array(&a, &b);
        assert_eq!(ans.index_item([]), 0.0);
    }
}

#[cfg(test)]
mod div {
    use crate::{
        device::cpu::Cpu,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    #[test]
    fn div_0d_0d() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![2.0], &[]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![3.0], &[]);
        let mut c = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros_like(&a);

        c.to_ref_mut().div_array(&a, &b);
        assert_eq!(c.as_slice(), &[2.0 / 3.0]);

        a.to_ref_mut().div_assign(&b);
        assert_eq!(a.as_slice(), &[2.0 / 3.0]);
    }

    #[test]
    fn div_1d_0d() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![2.0, 3.0], &[2]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![3.0], &[]);
        let mut c = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros_like(&a);
        c.to_ref_mut().div_array(&a, &b);
        assert_eq!(c.as_slice(), &[2.0 / 3.0, 3.0 / 3.0]);

        a.to_ref_mut().div_assign(&b);
        assert_eq!(a.as_slice(), &[2.0 / 3.0, 3.0 / 3.0]);
    }

    #[test]
    fn div_3d_3d() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
        );
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[2, 2, 2],
        );
        let mut c = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros_like(&a);
        c.to_ref_mut().div_array(&a, &b);
        let ans = vec![
            1.0 / 2.0,
            2.0 / 3.0,
            3.0 / 4.0,
            4.0 / 5.0,
            5.0 / 6.0,
            6.0 / 7.0,
            7.0 / 8.0,
            8.0 / 9.0,
        ];
        let ans = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(ans, &[2, 2, 2]);
        // let diff = c.to_ref() - ans.to_ref();
        // let asum = diff.asum();
        // assert_eq!(asum, 0.0);
        //
        // a.div_assign(b);
        // let diff = a.to_ref() - ans.to_view();
        // let asum = diff.asum();
        // assert_eq!(asum, 0.0);
    }

    #[test]
    fn div_4d_2d() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ],
            &[2, 2, 2, 2],
        );
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
        let mut c = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros_like(&a);
        c.to_ref_mut().div_array(&a, &b);
        let ans = vec![
            1.0 / 2.0,
            2.0 / 3.0,
            3.0 / 4.0,
            4.0 / 5.0,
            5.0 / 2.0,
            6.0 / 3.0,
            7.0 / 4.0,
            8.0 / 5.0,
            1.0 / 2.0,
            2.0 / 3.0,
            3.0 / 4.0,
            4.0 / 5.0,
            5.0 / 2.0,
            6.0 / 3.0,
            7.0 / 4.0,
            8.0 / 5.0,
        ];
        let ans = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(ans, &[2, 2, 2, 2]);
        // let diff = c.to_ref() - ans.to_view();
        // let asum = diff.asum();
        // assert_eq!(asum, 0.0);

        a.to_ref_mut().div_assign(&b);
        // let diff = a.to_ref() - ans.to_view();
        // let asum = diff.asum();
        // assert_eq!(asum, 0.0);
    }

    #[test]
    fn broadcast_4d_4d() {
        let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ],
            &[2, 2, 2, 2],
        );
        let b =
            Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[1, 1, 2, 2]);
        let mut c = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros_like(&a);
        c.to_ref_mut().div_array(&a, &b);
        let ans = vec![
            1.0 / 2.0,
            2.0 / 3.0,
            3.0 / 4.0,
            4.0 / 5.0,
            5.0 / 2.0,
            6.0 / 3.0,
            7.0 / 4.0,
            8.0 / 5.0,
            1.0 / 2.0,
            2.0 / 3.0,
            3.0 / 4.0,
            4.0 / 5.0,
            5.0 / 2.0,
            6.0 / 3.0,
            7.0 / 4.0,
            8.0 / 5.0,
        ];
        let ans = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(ans, &[2, 2, 2, 2]);
        // let diff = c - ans;
        // let asum = diff.asum();
        // assert_eq!(asum, 0.0);

        c.to_ref_mut().div_array(&b, &a);
        let ans = vec![
            2.0 / 1.0,
            3.0 / 2.0,
            4.0 / 3.0,
            5.0 / 4.0,
            2.0 / 5.0,
            3.0 / 6.0,
            4.0 / 7.0,
            5.0 / 8.0,
            2.0 / 1.0,
            3.0 / 2.0,
            4.0 / 3.0,
            5.0 / 4.0,
            2.0 / 5.0,
            3.0 / 6.0,
            4.0 / 7.0,
            5.0 / 8.0,
        ];
        let ans = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(ans, &[2, 2, 2, 2]);
        // let diff = c.to_ref() - ans.to_view();
        // let asum = diff.asum();
        // assert_eq!(asum, 0.0);
    }
}
#[cfg(test)]
mod mul {
    // use crate::{
    //     constructor::{ones::Ones, zeros::Zeros},
    //     matrix::{IndexItem, MatrixSlice, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
    //     matrix_impl::{OwnedMatrix0D, OwnedMatrix1D, OwnedMatrix2D, OwnedMatrix4D, Matrix::<Owned<f32>, DimDyn, Cpu>},
    //     operation::basic_operations::MatrixMul,
    //     slice,
    // };
    //
    // use super::MatrixSin;

    use crate::{
        device::cpu::Cpu,
        dim::DimDyn,
        matrix::{Matrix, Owned},
        slice_dynamic,
    };

    #[test]
    fn mul_1d_scalar() {
        let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![2.0], []);
        let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros([3]);
        ans.to_ref_mut().mul_array(&a, &b);

        assert_eq!(ans.index_item([0]), 2.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 6.0);
    }

    #[test]
    fn scalar_1d() {
        let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1., 2., 3.], [3]);
        let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros([3]);
        ans.to_ref_mut().mul_scalar(&a, 2.);

        assert_eq!(ans.index_item([0]), 2.);
        assert_eq!(ans.index_item([1]), 4.);
        assert_eq!(ans.index_item([2]), 6.);
    }

    #[test]
    fn sliced_scalar_1d() {
        let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1., 2., 3., 4.], [4]);
        let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros([2]);
        ans.to_ref_mut()
            .mul_scalar(&a.to_ref().slice(slice_dynamic!(..;2)), 2.);

        assert_eq!(ans.index_item([0]), 2.);
        assert_eq!(ans.index_item([1]), 6.);
    }

    #[test]
    fn scalar_2d() {
        let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros([2, 3]);
        ans.to_ref_mut().mul_scalar(&a, 2.);

        assert_eq!(ans.index_item([0, 0]), 2.);
        assert_eq!(ans.index_item([0, 1]), 4.);
        assert_eq!(ans.index_item([0, 2]), 6.);
        assert_eq!(ans.index_item([1, 0]), 8.);
        assert_eq!(ans.index_item([1, 1]), 10.);
        assert_eq!(ans.index_item([1, 2]), 12.);
    }

    #[test]
    fn default_1d_1d() {
        let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1., 2., 3.], [3]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1., 2., 3.], [3]);
        let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros([3]);
        ans.to_ref_mut().mul_array(&a, &b);

        assert_eq!(ans.index_item([0]), 1.);
        assert_eq!(ans.index_item([1]), 4.);
        assert_eq!(ans.index_item([2]), 9.);
    }

    #[test]
    fn sliced_1d_1d() {
        let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1., 2., 3., 4.], [4]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1., 2., 3., 4.], [4]);
        let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros([2]);
        ans.to_ref_mut().mul_array(
            &a.to_ref().slice(slice_dynamic!(..;2)),
            &b.to_ref().slice(slice_dynamic!(..;2)),
        );

        assert_eq!(ans.index_item([0]), 1.);
        assert_eq!(ans.index_item([1]), 9.);
    }

    #[test]
    fn default_2d_2d() {
        let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros([2, 3]);
        ans.to_ref_mut().mul_array(&a, &b);

        assert_eq!(ans.index_item([0, 0]), 1.);
        assert_eq!(ans.index_item([0, 1]), 4.);
        assert_eq!(ans.index_item([0, 2]), 9.);
        assert_eq!(ans.index_item([1, 0]), 16.);
        assert_eq!(ans.index_item([1, 1]), 25.);
        assert_eq!(ans.index_item([1, 2]), 36.);
    }

    #[test]
    fn sliced_4d_2d() {
        let mut a_vec = Vec::new();
        for i in 0..2 * 2 * 2 * 2 {
            a_vec.push(i as f32);
        }

        let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(a_vec, [2, 2, 2, 2]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1., 2.], [2]);

        let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros([2, 2, 2, 2]);

        ans.to_ref_mut().mul_array(&a, &b);

        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        assert_eq!(
                            ans.index_item([i, j, k, l]),
                            a.index_item([i, j, k, l]) * b.index_item([l])
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mul_4d_2d_dyn() {
        let ones_4d = Matrix::<Owned<f32>, DimDyn, Cpu>::ones([2, 2, 2, 2]);
        let ones_2d = Matrix::<Owned<f32>, DimDyn, Cpu>::ones([2, 2]);
        let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros([2, 2, 2, 2]);
        ans.to_ref_mut().mul_array(&ones_4d, &ones_2d);
    }

    #[test]
    fn default_0d_0d() {
        let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![10.], &[]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![20.], &[]);
        let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[]);
        ans.to_ref_mut().mul_array(&a, &b);
        assert_eq!(ans.index_item(&[]), 200.);
    }

    // #[test]
    // fn sin_0d() {
    //     let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1.0], &[]);
    //     let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[]);
    //     ans.sin(a.to_ref());
    //     let ans = ans.index_item(&[]);
    //     assert!(ans - 1.0_f32.sin() < 1e-6);
    // }
    //
    // #[test]
    // fn sin1d() {
    //     let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    //     let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[3]);
    //     ans.sin(a.to_ref());
    //     assert!(ans.index_item(&[0]) - 1.0_f32.sin() < 1e-6);
    //     assert!(ans.index_item(&[1]) - 2.0_f32.sin() < 1e-6);
    //     assert!(ans.index_item(&[2]) - 3.0_f32.sin() < 1e-6);
    // }
    //
    // #[test]
    // fn sin_2d() {
    //     let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    //     let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[2, 2]);
    //     ans.sin(a.to_ref());
    //     assert!(ans.index_item(&[0, 0]) - 1.0_f32.sin() < 1e-6);
    //     assert!(ans.index_item(&[0, 1]) - 2.0_f32.sin() < 1e-6);
    //     assert!(ans.index_item(&[1, 0]) - 3.0_f32.sin() < 1e-6);
    //     assert!(ans.index_item(&[1, 1]) - 4.0_f32.sin() < 1e-6);
    // }
    //
    // #[test]
    // fn sin_3d() {
    //     let a = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    //         &[2, 1, 3],
    //     );
    //     let mut ans = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[2, 1, 3]);
    //     ans.sin(a.to_ref());
    //     assert!(ans.index_item(&[0, 0, 0]) - 1.0_f32.sin() < 1e-6);
    //     assert!(ans.index_item(&[0, 0, 1]) - 2.0_f32.sin() < 1e-6);
    //     assert!(ans.index_item(&[0, 0, 2]) - 3.0_f32.sin() < 1e-6);
    //     assert!(ans.index_item(&[1, 0, 0]) - 4.0_f32.sin() < 1e-6);
    //     assert!(ans.index_item(&[1, 0, 1]) - 5.0_f32.sin() < 1e-6);
    //     assert!(ans.index_item(&[1, 0, 2]) - 6.0_f32.sin() < 1e-6);
    // }
}

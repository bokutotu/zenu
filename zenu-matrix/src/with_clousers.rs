use crate::{
    device::DeviceBase,
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
pub(crate) fn scalar_array_with_closure<T: Num, D: DeviceBase, B: Copy, F>(
    a: &Matrix<Ref<&mut T>, DimDyn, D>,
    b: B,
    f: F,
) where
    F: Fn(&Matrix<Ref<&mut T>, DimDyn, D>, B) + Copy,
{
    if a.shape_stride().is_default_stride() {
        f(a, b);
    } else {
        let num = a.shape()[0];
        for i in 0..num {
            scalar_array_with_closure(&a.index_axis_mut(Index0D::new(i)), b, f);
        }
    }
}

fn array_array_scalar_check<T: Num, D: DeviceBase>(
    a: &Matrix<Ref<&mut T>, DimDyn, D>,
    b: &Matrix<Ref<&T>, DimDyn, D>,
) {
    if a.shape() != b.shape() {
        panic!("The shape of the matrix is different.");
    }
}
fn array_array_scalar_with_closure<T: Num, D: DeviceBase, B: Copy, F>(
    a: &Matrix<Ref<&mut T>, DimDyn, D>,
    b: &Matrix<Ref<&T>, DimDyn, D>,
    c: B,
    f: F,
) where
    F: Fn(&Matrix<Ref<&mut T>, DimDyn, D>, &Matrix<Ref<&T>, DimDyn, D>, B) + Copy,
{
    if a.shape_stride().is_default_stride() && b.shape_stride().is_default_stride() {
        f(a, b, c);
    } else {
        let num = a.shape()[0];
        for i in 0..num {
            array_array_scalar_with_closure(
                &a.index_axis_mut(Index0D::new(i)),
                &b.index_axis(Index0D::new(i)),
                c,
                f,
            );
        }
    }
}
pub(crate) fn array_array_scalar<T, D, F>(
    a: &Matrix<Ref<&mut T>, DimDyn, D>,
    b: &Matrix<Ref<&T>, DimDyn, D>,
    c: T,
    f: F,
) where
    T: Num,
    D: DeviceBase,
    F: Fn(&Matrix<Ref<&mut T>, DimDyn, D>, &Matrix<Ref<&T>, DimDyn, D>, T) + Copy,
{
    array_array_scalar_check(a, b);
    array_array_scalar_with_closure(a, b, c, f);
}

fn array_array_check<T: Num, D: DeviceBase>(
    a: &Matrix<Ref<&mut T>, DimDyn, D>,
    b: &Matrix<Ref<&T>, DimDyn, D>,
) {
    if a.shape().len() < b.shape().len() {
        panic!("The rank of the matrix is different.");
    }
    if !(a.shape().is_include(b.shape()) || a.shape().is_include_bradcast(b.shape())) {
        panic!(
            "other shape is not include self shape {:?} {:?}",
            a.shape(),
            b.shape()
        );
    }
    if !a.shape().is_include_bradcast(b.shape()) {
        panic!(
            "other shape is not include self shape {:?} {:?}",
            a.shape(),
            b.shape()
        );
    }
}
fn array_array_with_closure<T: Num, D: DeviceBase, FMatMat, FMatSca>(
    a: &Matrix<Ref<&mut T>, DimDyn, D>,
    b: &Matrix<Ref<&T>, DimDyn, D>,
    f_mat_mat: FMatMat,
    f_mat_scalar_ptr: FMatSca,
) where
    FMatMat: Fn(&Matrix<Ref<&mut T>, DimDyn, D>, &Matrix<Ref<&T>, DimDyn, D>) + Copy,
    FMatSca: Fn(&Matrix<Ref<&mut T>, DimDyn, D>, *const T) + Copy,
{
    if a.shape().is_scalar() {
        f_mat_scalar_ptr(a, b.as_ptr());
    } else if b.shape().is_scalar() && a.shape_stride().is_default_stride() {
        f_mat_scalar_ptr(a, b.as_ptr());
    } else if a.shape_stride().is_default_stride()
        && b.shape_stride().is_default_stride()
        && a.shape() == b.shape()
    {
        f_mat_mat(a, b);
    } else {
        let num = a.shape()[0];
        for i in 0..num {
            let b = get_tmp_matrix(b, a.shape().len(), i, b.shape().len());
            array_array_with_closure(
                &a.index_axis_mut(Index0D::new(i)),
                &b,
                f_mat_mat,
                f_mat_scalar_ptr,
            );
        }
    }
}
pub(crate) fn array_array<T, D, FMatMat, FMatSca>(
    a: &Matrix<Ref<&mut T>, DimDyn, D>,
    b: &Matrix<Ref<&T>, DimDyn, D>,
    f_mat_mat: FMatMat,
    f_mat_scalar_ptr: FMatSca,
) where
    T: Num,
    D: DeviceBase,
    FMatMat: Fn(&Matrix<Ref<&mut T>, DimDyn, D>, &Matrix<Ref<&T>, DimDyn, D>) + Copy,
    FMatSca: Fn(&Matrix<Ref<&mut T>, DimDyn, D>, *const T) + Copy,
{
    array_array_check(a, b);
    array_array_with_closure(a, b, f_mat_mat, f_mat_scalar_ptr);
}

fn array_array_array_check<T: Num, D: DeviceBase>(
    self_: &Matrix<Ref<&mut T>, DimDyn, D>,
    lhs: &Matrix<Ref<&T>, DimDyn, D>,
    rhs: &Matrix<Ref<&T>, DimDyn, D>,
) {
    let larger_dim = larger_shape(lhs.shape(), rhs.shape());
    let smaller_dim = smaller_shape(lhs.shape(), rhs.shape());
    if !(larger_dim.is_include(smaller_dim)
        || DimDyn::from(self_.shape().slice()).is_include_bradcast(smaller_dim))
    {
        panic!(
            "self dim is not match other dims self dim {:?}, lhs dim {:?} rhs dim {:?}",
            self_.shape(),
            lhs.shape(),
            rhs.shape()
        );
    }
    if self_.shape().slice() != larger_dim.slice() && self_.shape().slice() != smaller_dim.slice() {
        panic!("longer shape lhs or rhs is same shape to self\n self.shape = {:?}\n lhs.shape() = {:?} \n rhs.shape() = {:?}",
            self_.shape(), lhs.shape(), rhs.shape());
    }
}
fn array_array_array_with_closure<T, D, FMatMat, FMatSca>(
    a: &Matrix<Ref<&mut T>, DimDyn, D>,
    b: &Matrix<Ref<&T>, DimDyn, D>,
    c: &Matrix<Ref<&T>, DimDyn, D>,
    f_mat_mat: FMatMat,
    f_mat_scalar_ptr: FMatSca,
) where
    T: Num,
    D: DeviceBase,
    FMatMat: Fn(
            &Matrix<Ref<&mut T>, DimDyn, D>,
            &Matrix<Ref<&T>, DimDyn, D>,
            &Matrix<Ref<&T>, DimDyn, D>,
        ) + Copy,
    FMatSca: Fn(&Matrix<Ref<&mut T>, DimDyn, D>, &Matrix<Ref<&T>, DimDyn, D>, *const T) + Copy,
{
    if b.shape().is_scalar()
        && a.shape_stride().is_default_stride()
        && c.shape_stride().is_default_stride()
        && a.shape() == c.shape()
    {
        f_mat_scalar_ptr(a, c, b.as_ptr());
    } else if c.shape().is_scalar()
        && a.shape_stride().is_default_stride()
        && b.shape_stride().is_default_stride()
        && a.shape() == b.shape()
    {
        f_mat_scalar_ptr(a, b, c.as_ptr());
    } else if a.shape_stride().is_default_stride()
        && b.shape_stride().is_default_stride()
        && c.shape_stride().is_default_stride()
        && a.shape() == b.shape()
        && a.shape() == c.shape()
    {
        f_mat_mat(a, b, c);
    } else {
        let num = a.shape()[0];
        for i in 0..num {
            let b = get_tmp_matrix(b, a.shape().len(), i, b.shape().len());
            let c = get_tmp_matrix(c, a.shape().len(), i, c.shape().len());
            array_array_array_with_closure(
                &a.index_axis_mut(Index0D::new(i)),
                &b,
                &c,
                f_mat_mat,
                f_mat_scalar_ptr,
            );
        }
    }
}
pub(crate) fn array_array_array<T, D, FMatMat, FMatSca>(
    a: &Matrix<Ref<&mut T>, DimDyn, D>,
    b: &Matrix<Ref<&T>, DimDyn, D>,
    c: &Matrix<Ref<&T>, DimDyn, D>,
    f_mat_mat: FMatMat,
    f_mat_scalar_ptr: FMatSca,
) where
    T: Num,
    D: DeviceBase,
    FMatMat: Fn(
            &Matrix<Ref<&mut T>, DimDyn, D>,
            &Matrix<Ref<&T>, DimDyn, D>,
            &Matrix<Ref<&T>, DimDyn, D>,
        ) + Copy,
    FMatSca: Fn(&Matrix<Ref<&mut T>, DimDyn, D>, &Matrix<Ref<&T>, DimDyn, D>, *const T) + Copy,
{
    array_array_array_check(a, b, c);
    array_array_array_with_closure(a, b, c, f_mat_mat, f_mat_scalar_ptr);
}

#[cfg(test)]
mod basic_ops {
    use std::{cell::RefCell, rc::Rc};

    use crate::{
        device::{cpu::Cpu, DeviceBase},
        dim::{DimDyn, DimTrait},
        matrix::{Matrix, Owned, Ref},
        num::Num,
        operation::basic_operations::AddOps,
        slice_dynamic,
    };

    use super::{array_array, array_array_array, array_array_scalar, scalar_array_with_closure};

    // test impl for Add
    impl<T, D> Matrix<Ref<&mut T>, DimDyn, D>
    where
        T: Num,
        D: DeviceBase + AddOps,
    {
        fn add_scalar_(&self, other: &Matrix<Ref<&T>, DimDyn, D>, scalar: T) {
            array_array_scalar(self, other, scalar, |a, b, c| {
                let len = a.shape().len();
                let num_elm = a.shape().num_elm();
                let to_stride = a.stride()[len - 1];
                let rhs_stride = b.stride()[len - 1];
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
        fn add_scalar_assign_(&self, scalar: T) {
            scalar_array_with_closure(self, scalar, |a, b| {
                let len = a.shape().len();
                let num_elm = a.shape().num_elm();
                let stride = a.stride()[len - 1];
                D::scalar_assign(a.as_mut_ptr(), b, num_elm, stride);
            });
        }

        fn add_assign_(&self, other: Matrix<Ref<&T>, DimDyn, D>) {
            array_array(
                self,
                &other,
                |a, b| {
                    let len = a.shape().len();
                    let num_elm = a.shape().num_elm();
                    let to_stride = a.stride()[len - 1];
                    let rhs_stride = b.stride()[len - 1];
                    D::array_assign(a.as_mut_ptr(), b.as_ptr(), num_elm, to_stride, rhs_stride);
                },
                |a, b| {
                    let len = a.shape().len();
                    let num_elm = a.shape().num_elm();
                    let stride = a.stride()[len - 1];
                    D::scalar_assign_ptr(a.as_mut_ptr(), b, num_elm, stride);
                },
            );
        }

        fn add_(&self, lhs: &Matrix<Ref<&T>, DimDyn, D>, rhs: &Matrix<Ref<&T>, DimDyn, D>) {
            array_array_array(
                self,
                lhs,
                rhs,
                |a, b, c| {
                    let len = a.shape().len();
                    let num_elm = a.shape().num_elm();
                    let to_stride = a.stride()[len - 1];
                    let lhs_stride = b.stride()[len - 1];
                    let rhs_stride = c.stride()[len - 1];
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
                    let len = a.shape().len();
                    let num_elm = a.shape().num_elm();
                    let to_stride = a.stride()[len - 1];
                    let lhs_stride = b.stride()[len - 1];
                    D::scalar_ptr(
                        a.as_mut_ptr(),
                        b.as_ptr(),
                        c,
                        num_elm,
                        to_stride,
                        lhs_stride,
                    );
                },
            );
        }
    }

    #[test]
    fn scalar_array_default_stride() {
        let num_called = Rc::new(RefCell::new(0));
        let clsuer = |_a: &Matrix<Ref<&mut f32>, DimDyn, Cpu>, _b: f32| {
            *num_called.borrow_mut() += 1;
        };

        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[10, 10, 10]);
        scalar_array_with_closure(&a.to_ref_mut().into_dyn_dim(), 1.0, &clsuer);
        assert_eq!(num_called.take(), 1);
    }

    #[test]
    fn scalar_array_sliced() {
        let num_called = Rc::new(RefCell::new(0));
        let clsuer = |_a: &Matrix<Ref<&mut f32>, DimDyn, Cpu>, _b: f32| {
            *num_called.borrow_mut() += 1;
        };

        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[10, 10, 10]);
        let a = a.to_ref_mut().slice_mut(slice_dynamic![.., ..;2, ..]);
        scalar_array_with_closure(&a.into_dyn_dim(), 1.0, &clsuer);
        assert_eq!(num_called.take(), 50);
    }

    #[test]
    fn add_scalar_default_stride() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[10, 10, 10]);
        a.to_ref_mut().add_scalar_assign_(1.0);
        assert_eq!(a.to_vec(), &[1.0; 10 * 10 * 10]);
    }

    #[test]
    fn add_scalar_sliced() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[2, 4, 5]);
        {
            let a_ref = a.to_ref_mut().slice_mut(slice_dynamic![.., ..;2, ..]);
            a_ref.add_scalar_assign_(1.0);
        }
        let mut ans = Vec::new();
        for _i in 0..2 {
            for j in 0..4 {
                for _k in 0..5 {
                    if j % 2 == 0 {
                        ans.push(1.0);
                    } else {
                        ans.push(0.0);
                    }
                }
            }
        }
        assert_eq!(a.to_vec(), ans);
    }

    #[test]
    fn add_assign_default_stride_same_shape() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[10, 10, 10]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1.0; 10 * 10 * 10], &[10, 10, 10]);
        a.to_ref_mut().add_assign_(b.to_ref());
        assert_eq!(a.to_vec(), &[1.0; 10 * 10 * 10]);
    }

    #[test]
    fn add_assign_default_stride_bradcast() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[10, 10, 10]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1.0; 10 * 10], &[10, 10]);
        a.to_ref_mut().add_assign_(b.to_ref());
        assert_eq!(a.to_vec(), &[1.0; 10 * 10 * 10]);
    }

    #[test]
    fn add_assign_other_stride() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[10, 10, 10]);
        let b = (0..10 * 10 * 10).map(|x| x as f32).collect::<Vec<f32>>();
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(b, &[10, 10, 10]);
        let b = b.slice(slice_dynamic![.., 2, ..]);
        a.to_ref_mut().add_assign_(b);
        let mut ans = Vec::new();
        for i in 0..10 {
            for j in 0..5 {
                for k in 0..10 {
                    let idx = i * 10 * 10 + j * 10 + k;
                    if j == 2 {
                        ans.push(idx as f32);
                    }
                }
            }
        }
        let ans = [&ans; 10].iter().cloned().flatten().collect::<Vec<&f32>>();
        let ans = ans.iter().map(|x| **x).collect::<Vec<f32>>();
        assert_eq!(a.to_vec(), ans);
    }

    #[test]
    fn add_assign_self_strided() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[20, 10, 10]);
        let b_vec = (0..10 * 10).map(|x| x as f32).collect::<Vec<f32>>();
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(b_vec.clone(), &[10, 10]);
        {
            let a_ref = a.to_ref_mut().slice_mut(slice_dynamic![..;2, .., ..]);
            a_ref.add_assign_(b.to_ref());
        }
        let mut ans = Vec::new();
        for i in 0..20 {
            if i % 2 == 0 {
                ans.extend_from_slice(&b_vec);
            } else {
                ans.extend_from_slice(&vec![0.0; 10 * 10]);
            }
        }
        assert_eq!(a.to_vec(), ans);
    }

    #[test]
    fn add_scalar_default_stride_() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[10, 10, 10]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1.0; 10 * 10 * 10], &[10, 10, 10]);
        a.to_ref_mut().add_scalar_(&b.to_ref(), 2.0);
        assert_eq!(a.to_vec(), &[3.0; 10 * 10 * 10]);
    }

    #[test]
    fn add_scalar_sliced_() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[2, 4, 5]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1.0; 2 * 4 * 5], &[2, 4, 5]);
        {
            let a_ref = a.to_ref_mut().slice_mut(slice_dynamic![.., ..;2, ..]);
            let b_ref = b.slice(slice_dynamic![.., ..;2, ..]);
            a_ref.add_scalar_(&b_ref, 2.0);
        }
        let mut ans = Vec::new();
        for _i in 0..2 {
            for j in 0..4 {
                for _k in 0..5 {
                    if j % 2 == 0 {
                        ans.push(3.0);
                    } else {
                        ans.push(0.0);
                    }
                }
            }
        }
        assert_eq!(a.to_vec(), ans);
    }

    #[test]
    fn add_default_stride_same_shape() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[10, 10, 10]);
        let b = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1.0; 10 * 10 * 10], &[10, 10, 10]);
        let c = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![2.0; 10 * 10 * 10], &[10, 10, 10]);
        a.to_ref_mut().add_(&b.to_ref(), &c.to_ref());
        assert_eq!(a.to_vec(), &[3.0; 10 * 10 * 10]);
    }
}

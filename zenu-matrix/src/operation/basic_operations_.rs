use crate::{
    device::DeviceBase,
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{Matrix, Ref, Repr},
    num::Num,
};

use super::basic_operations::AddOps;

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

fn scalar_array_with_closure<T: Num, D: DeviceBase, B: Copy, F>(
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

fn array_array_with_closure<T: Num, D: DeviceBase, FMatMat, FMatSca>(
    a: &Matrix<Ref<&mut T>, DimDyn, D>,
    b: &Matrix<Ref<&T>, DimDyn, D>,
    f_mat_mat: FMatMat,
    f_mat_scalar_ptr: FMatSca,
) where
    FMatMat: Fn(&Matrix<Ref<&mut T>, DimDyn, D>, &Matrix<Ref<&T>, DimDyn, D>) + Copy,
    FMatSca: Fn(&Matrix<Ref<&mut T>, DimDyn, D>, *const T) + Copy,
{
    if a.shape_stride().is_default_stride()
        && b.shape_stride().is_default_stride()
        && a.shape() == b.shape()
    {
        f_mat_mat(a, b);
    } else if b.shape().is_empty() && a.shape_stride().is_default_stride() {
        f_mat_scalar_ptr(a, b.as_ptr());
    } else {
        let num = a.shape()[0];
        for i in 0..num {
            let b = get_tmp_matrix(b, num, i, a.shape()[0]);
            array_array_with_closure(
                &a.index_axis_mut(Index0D::new(i)),
                &b,
                f_mat_mat,
                f_mat_scalar_ptr,
            );
        }
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
    if a.shape_stride().is_default_stride()
        && b.shape_stride().is_default_stride()
        && c.shape_stride().is_default_stride()
        && a.shape() == b.shape()
        && a.shape() == c.shape()
    {
        f_mat_mat(a, b, c);
    } else if b.shape().is_empty()
        && a.shape_stride().is_default_stride()
        && c.shape_stride().is_default_stride()
        && a.shape() == c.shape()
    {
        f_mat_scalar_ptr(a, c, b.as_ptr());
    } else if c.shape().is_empty()
        && a.shape_stride().is_default_stride()
        && b.shape_stride().is_default_stride()
        && a.shape() == b.shape()
    {
        f_mat_scalar_ptr(a, b, c.as_ptr());
    } else {
        let num = a.shape()[0];
        for i in 0..num {
            let b = get_tmp_matrix(b, num, i, a.shape()[0]);
            let c = get_tmp_matrix(c, num, i, a.shape()[0]);
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

// test impl for Add
impl<T, D> Matrix<Ref<&mut T>, DimDyn, D>
where
    T: Num,
    D: DeviceBase + AddOps,
{
    pub fn add_scalar_(self, scalar: T) {
        scalar_array_with_closure(&self.into_dyn_dim(), scalar, |a, b| {
            let len = a.shape().len();
            let num_elm = a.shape().num_elm();
            let stride = a.stride()[len - 1];
            D::scalar_assign(a.as_mut_ptr(), b, num_elm, stride);
        });
    }
}

#[cfg(test)]
mod basic_ops {
    use std::{cell::RefCell, rc::Rc};

    use crate::{
        device::cpu::Cpu,
        dim::DimDyn,
        matrix::{Matrix, Owned, Ref},
        operation::basic_operations_::scalar_array_with_closure,
        slice_dynamic,
    };

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
        a.to_ref_mut().add_scalar_(1.0);
        assert_eq!(a.to_vec(), &[1.0; 10 * 10 * 10]);
    }

    #[test]
    fn add_scalar_sliced() {
        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[2, 4, 5]);
        let a_ref = a.to_ref_mut().slice_mut(slice_dynamic![.., ..;2, ..]);
        a_ref.add_scalar_(1.0);
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
}

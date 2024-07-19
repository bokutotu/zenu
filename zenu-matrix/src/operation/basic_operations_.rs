use crate::{
    device::DeviceBase,
    dim::{DimDyn, DimTrait},
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

fn is_1d_1(a: &[usize]) -> bool {
    a[0] == 1
}

struct ArrayAssignParams {
    offset: usize,
    num_elm: usize,
    stride: usize,
}

fn scalar_array_with_closure<T: Num, D: DeviceBase, B: Copy>(
    a: &Matrix<Ref<&mut T>, DimDyn, D>,
    b: B,
    f: &mut impl FnMut(&Matrix<Ref<&mut T>, DimDyn, D>, B),
) {
    println!("a.shape_stride(): {:?}", a.shape_stride());
    if a.shape_stride().is_default_stride() {
        f(a, b);
    } else {
        let num = a.shape()[0];
        for i in 0..num {
            scalar_array_with_closure(&a.index_axis_mut(Index0D::new(i)), b, f);
        }
    }
}

#[cfg(test)]
mod basic_ops {
    use crate::{
        device::cpu::Cpu,
        dim::DimDyn,
        matrix::{Matrix, Owned, Ref},
        operation::basic_operations_::scalar_array_with_closure,
        slice_dynamic,
    };

    #[test]
    fn scalar_array_default_stride() {
        let mut num_called = 0;
        let mut clsuer = |_a: &Matrix<Ref<&mut f32>, DimDyn, Cpu>, _b: f32| {
            num_called += 1;
        };

        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[10, 10, 10]);
        scalar_array_with_closure(&a.to_ref_mut().into_dyn_dim(), 1.0, &mut clsuer);
        assert_eq!(num_called, 1);
    }

    #[test]
    fn scalar_array_sliced() {
        let mut num_called = 0;
        let mut clsuer = |_a: &Matrix<Ref<&mut f32>, DimDyn, Cpu>, _b: f32| {
            num_called += 1;
        };

        let mut a = Matrix::<Owned<f32>, DimDyn, Cpu>::zeros(&[10, 10, 10]);
        let a = a.to_ref_mut().slice_mut(slice_dynamic![.., ..;2, ..]);
        scalar_array_with_closure(&a.into_dyn_dim(), 1.0, &mut clsuer);
        assert_eq!(num_called, 50);
    }
}

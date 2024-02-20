use crate::{
    dim::{Dim1, DimTrait},
    index::Index0D,
    matrix::{IndexAxisDyn, IndexAxisMutDyn, IndexItem, IndexItemAsign, MatrixBase, ViewMutMatix},
    matrix_impl::{matrix_into_dim, Matrix},
    memory::{ViewMemory, ViewMutMemory},
    num::Num,
    operation::copy_from::CopyFrom,
};

fn add_assign_matrix_scalar<T, LM, D>(to: Matrix<LM, D>, other: T)
where
    T: Num,
    LM: ViewMutMemory<Item = T>,
    D: DimTrait,
{
    let mut to = to.into_dyn_dim();

    match to.shape().slice() {
        [] => to.index_item_asign(&[] as &[usize], to.index_item(&[] as &[usize]) + other),
        [a] => {
            for i in 0..*a {
                to.index_item_asign([i], to.index_item([i]) + other);
            }
        }
        [a, _] => {
            for i in 0..*a {
                let to = to.index_axis_mut_dyn(Index0D::new(i));
                add_assign_matrix_scalar(to, other);
            }
        }
        [a, _, _] => {
            for i in 0..*a {
                let to = to.index_axis_mut_dyn(Index0D::new(i));
                add_assign_matrix_scalar(to, other);
            }
        }
        [a, _, _, _] => {
            for i in 0..*a {
                let to = to.index_axis_mut_dyn(Index0D::new(i));
                add_assign_matrix_scalar(to, other);
            }
        }
        _ => panic!("not implemented: this is bug. please report this bug."),
    }
}

// matrix_add_scalar_assignを使用して,
// add_matrix_scalarを実装する
fn add_matrix_scalar<T, SM, LM, D>(to: Matrix<SM, D>, lhs: Matrix<LM, D>, rhs: T)
where
    T: Num,
    SM: ViewMutMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    D: DimTrait,
{
    assert_eq!(to.shape(), lhs.shape());
    let mut to = to.into_dyn_dim();
    let lhs = lhs.into_dyn_dim();
    to.copy_from(&lhs);
    add_assign_matrix_scalar(to, rhs);
}

fn add_assign_matrix_matrix<T, VM, V, D1, D2>(source: Matrix<VM, D1>, other: Matrix<V, D2>)
where
    T: Num,
    VM: ViewMutMemory<Item = T>,
    V: ViewMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
{
    let mut source = source.into_dyn_dim();
    let other = other.into_dyn_dim();

    assert!(source.shape().is_include(&other.shape()));

    if source.shape().is_empty() {
        source.index_item_asign(
            &[] as &[usize],
            source.index_item(&[] as &[usize]) + other.index_item(&[] as &[usize]),
        );
        return;
    }
    if other.shape().is_empty() {
        let scalar = other.index_item(&[] as &[usize]);
        add_assign_matrix_scalar(source, scalar);
        return;
    }
    //

    if source.shape().len() == 1 {
        let mut source: Matrix<VM, Dim1> = matrix_into_dim(source);
        for i in 0..source.shape()[0] {
            let s_itm = source.index_item([i]);
            let o_itm = other.index_item([i]);
            source.index_item_asign([i], s_itm + o_itm);
        }
    } else if source.shape() == other.shape() {
        let mut source = source;
        for i in 0..source.shape()[0] {
            let source = source.index_axis_mut_dyn(Index0D::new(i));
            let other = other.index_axis_dyn(Index0D::new(i));
            add_assign_matrix_matrix(source, other);
        }
    } else {
        let mut source = source;
        for i in 0..source.shape()[0] {
            let source = source.index_axis_mut_dyn(Index0D::new(i));
            let other = other.clone();
            add_assign_matrix_matrix(source, other);
        }
    }
}

fn add_matrix_matrix<T, VM, L, R, D1, D2, D3>(
    to: Matrix<VM, D1>,
    lhs: Matrix<L, D2>,
    rhs: Matrix<R, D3>,
) where
    T: Num,
    VM: ViewMutMemory<Item = T>,
    L: ViewMemory<Item = T>,
    R: ViewMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
    D3: DimTrait,
{
    assert_eq!(to.shape().slice(), lhs.shape().slice());
    let mut to = to.into_dyn_dim();
    let lhs = lhs.into_dyn_dim();
    to.copy_from(&lhs);
    add_assign_matrix_matrix(to, rhs);
}
pub trait MatrixAdd<Rhs, Lhs>: ViewMutMatix + MatrixBase {
    fn add(self, lhs: Rhs, rhs: Lhs);
}

pub trait MatrixAddAssign<Rhs>: ViewMutMatix + MatrixBase {
    fn add_assign(self, rhs: Rhs);
}

// matrix add scalar
impl<T, RM, SM, D> MatrixAdd<Matrix<RM, D>, T> for Matrix<SM, D>
where
    T: Num,
    RM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
    D: DimTrait,
{
    fn add(self, lhs: Matrix<RM, D>, rhs: T) {
        add_matrix_scalar(self, lhs, rhs);
    }
}

impl<T, RM, LM, SM, D1, D2> MatrixAdd<Matrix<LM, D1>, Matrix<RM, D2>> for Matrix<SM, D1>
where
    T: Num,
    RM: ViewMemory<Item = T>,
    LM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
{
    fn add(self, lhs: Matrix<LM, D1>, rhs: Matrix<RM, D2>) {
        add_matrix_matrix(self, lhs, rhs);
    }
}

impl<T, SM, D> MatrixAddAssign<T> for Matrix<SM, D>
where
    T: Num,
    SM: ViewMutMemory<Item = T>,
    D: DimTrait,
{
    fn add_assign(self, rhs: T) {
        add_assign_matrix_scalar(self, rhs);
    }
}

impl<T, LM, SM, D1, D2> MatrixAddAssign<Matrix<LM, D1>> for Matrix<SM, D2>
where
    T: Num,
    LM: ViewMemory<Item = T>,
    SM: ViewMutMemory<Item = T>,
    D1: DimTrait,
    D2: DimTrait,
{
    fn add_assign(self, rhs: Matrix<LM, D1>) {
        add_assign_matrix_matrix(self, rhs);
    }
}

#[cfg(test)]
mod add {
    use crate::{
        matrix::{MatrixSlice, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::{CpuOwnedMatrix0D, CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix3D},
        operation::zeros::Zeros,
        slice,
    };

    use super::*;

    #[test]
    fn add_dyn_dyn() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let ans = CpuOwnedMatrix1D::<f32>::zeros([3]);

        let a = a.into_dyn_dim();
        let b = b.into_dyn_dim();
        let mut ans = ans.into_dyn_dim();

        ans.to_view_mut().add(a.to_view(), b.to_view());
    }

    #[test]
    fn add_1d_scalar() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros([3]);
        let b = CpuOwnedMatrix0D::from_vec(vec![2.0], []);
        ans.to_view_mut().add(a.to_view(), b.to_view());

        assert_eq!(ans.index_item([0]), 3.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 5.0);
    }

    #[test]
    fn add_1d_scalar_default_stride() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros([3]);
        ans.to_view_mut().add(a.to_view(), 1.0);

        assert_eq!(ans.index_item([0]), 2.0);
        assert_eq!(ans.index_item([1]), 3.0);
        assert_eq!(ans.index_item([2]), 4.0);
    }

    #[test]
    fn add_1d_scalar_sliced() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros([3]);

        let sliced = a.slice(slice!(..;2));

        ans.to_view_mut().add(sliced.to_view(), 1.0);

        assert_eq!(ans.index_item([0]), 2.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 6.0);
    }

    #[test]
    fn add_3d_scalar_sliced() {
        let a = CpuOwnedMatrix3D::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
            ],
            [3, 3, 4],
        );

        let mut ans = CpuOwnedMatrix3D::<f32>::zeros([3, 3, 2]);

        let sliced = a.slice(slice!(.., .., ..;2));

        ans.to_view_mut().add(sliced.to_view(), 1.0);

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
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let mut ans = CpuOwnedMatrix1D::<f32>::zeros([3]);
        ans.to_view_mut().add(a.to_view(), b.to_view());

        assert_eq!(ans.index_item([0]), 2.0);
        assert_eq!(ans.index_item([1]), 4.0);
        assert_eq!(ans.index_item([2]), 6.0);
    }

    #[test]
    fn add_1d_1d_sliced() {
        let a = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);
        let b = CpuOwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]);

        let mut ans = CpuOwnedMatrix1D::<f32>::zeros([3]);

        let sliced_a = a.slice(slice!(..;2));
        let sliced_b = b.slice(slice!(1..;2));

        ans.to_view_mut()
            .add(sliced_a.to_view(), sliced_b.to_view());

        assert_eq!(ans.index_item([0]), 3.0);
        assert_eq!(ans.index_item([1]), 7.0);
        assert_eq!(ans.index_item([2]), 11.0);
    }

    #[test]
    fn add_2d_1d_default() {
        let a = CpuOwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );

        let b = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], [8]);

        let mut ans = CpuOwnedMatrix2D::<f32>::zeros([2, 2]);

        let sliced_a = a.slice(slice!(..2, ..2));
        let sliced_b = b.slice(slice!(..2));

        ans.to_view_mut()
            .add(sliced_a.to_view(), sliced_b.to_view());

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
        let a = CpuOwnedMatrix3D::from_vec(v, [4, 4, 4]);

        let b = CpuOwnedMatrix1D::from_vec(vec![1., 2., 3., 4.], [4]);

        let mut ans = CpuOwnedMatrix3D::<f32>::zeros([2, 2, 2]);

        let sliced_a = a.slice(slice!(..2, 1..;2, ..2));
        let sliced_b = b.slice(slice!(..2));

        ans.to_view_mut()
            .add(sliced_a.to_view(), sliced_b.to_view());

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
        let a = CpuOwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );

        let b = CpuOwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
            [4, 4],
        );

        let mut ans = CpuOwnedMatrix2D::<f32>::zeros([4, 4]);
        ans.to_view_mut().add(a.to_view(), b.to_view());

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
}

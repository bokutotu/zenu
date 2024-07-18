use crate::{
    device::DeviceBase,
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{Matrix, Ref, Repr},
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
/// Matrixとスカラのオペレーションをする際、DeviceのAddなどのOperationを行う際、
/// 最初のptrからのoffset, num_elm, strideのリストを返す
fn get_array_assign_prams<R: Repr, D: DeviceBase>(
    mat: Matrix<R, DimDyn, D>,
) -> Vec<ArrayAssignParams> {
    if mat.shape_stride().is_default_stride() {
        return vec![ArrayAssignParams {
            offset: 0,
            num_elm: mat.shape().num_elm(),
            stride: 1,
        }];
    }
}

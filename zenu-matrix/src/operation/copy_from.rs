use crate::{
    blas::Blas,
    dim::{DimDyn, DimTrait},
    matrix::{AsMutPtr, AsPtr, MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::{Memory, ToViewMemory, ToViewMutMemory},
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
    shape_stride::ShapeStride,
};

pub trait CopyFrom<RHS>: ToViewMutMatrix
where
    RHS: ToViewMatrix,
{
    fn copy_from(&mut self, rhs: &RHS);
}

impl<T, V, VM> CopyFrom<Matrix<V, DimDyn>> for Matrix<VM, DimDyn>
where
    T: Num,
    VM: ToViewMutMemory<Item = T>,
    V: ToViewMemory<Item = T>,
{
    fn copy_from(&mut self, rhs: &Matrix<V, DimDyn>) {
        assert_eq!(self.shape().slice(), rhs.shape().slice(), "Shape mismatch");
        copy(self.to_view_mut(), rhs.to_view());
    }
}

fn generate_combinations(dimensions: &[usize]) -> Vec<Vec<usize>> {
    let mut combinations = Vec::new();
    let mut current_combination = vec![0; dimensions.len()];

    fn backtrack(
        dimensions: &[usize],
        current_combination: &mut Vec<usize>,
        current_index: usize,
        combinations: &mut Vec<Vec<usize>>,
    ) {
        if current_index == dimensions.len() {
            combinations.push(current_combination.clone());
            return;
        }

        for i in 0..dimensions[current_index] {
            current_combination[current_index] = i;
            backtrack(
                dimensions,
                current_combination,
                current_index + 1,
                combinations,
            );
        }
    }

    backtrack(dimensions, &mut current_combination, 0, &mut combinations);
    combinations
}

fn copy_apply_index(shape: DimDyn) -> Vec<DimDyn> {
    let slice = shape.slice();
    let combinations = generate_combinations(&slice);
    combinations
        .into_iter()
        .map(|combination| DimDyn::from(&combination as &[usize]))
        .collect()
}

/// blasをそのまま適応できる最大のshapeのindex(後ろから数えた場合)を返す
fn get_max_shape_idx_of_apply_blas(a: ShapeStride<DimDyn>, b: ShapeStride<DimDyn>) -> usize {
    let mut idx = 1;
    let min_len = std::cmp::min(a.shape().len(), b.shape().len());
    let a_len = a.shape().len();
    let b_len = b.shape().len();

    if min_len == 1 {
        return 1;
    }
    if min_len == 0 {
        return 0;
    }
    if min_len == 2 {
        let a_shape_part = DimDyn::from(&a.shape().slice()[a_len - 2..]);
        let b_shape_part = DimDyn::from(&b.shape().slice()[b_len - 2..]);
        let a_stride_part = DimDyn::from(&a.stride().slice()[a_len - 2..]);
        let b_stride_part = DimDyn::from(&b.stride().slice()[b_len - 2..]);
        let a_part_shape_stride = ShapeStride::new(a_shape_part, a_stride_part);
        let b_part_shape_stride = ShapeStride::new(b_shape_part, b_stride_part);

        if !(a_part_shape_stride.is_transposed() || b_part_shape_stride.is_transposed())
            && a_part_shape_stride.is_contiguous()
            && b_part_shape_stride.is_contiguous()
        {
            return 2;
        } else {
            return 1;
        }
    }

    for i in 2..=min_len {
        let a_shape_part = DimDyn::from(&a.shape().slice()[a_len - i..]);
        let b_shape_part = DimDyn::from(&b.shape().slice()[b_len - i..]);
        let a_stride_part = DimDyn::from(&a.stride().slice()[a_len - i..]);
        let b_stride_part = DimDyn::from(&b.stride().slice()[b_len - i..]);
        let a_part_shape_stride = ShapeStride::new(a_shape_part, a_stride_part);
        let b_part_shape_stride = ShapeStride::new(b_shape_part, b_stride_part);

        if !a_part_shape_stride.is_transposed()
            && (a_part_shape_stride.is_transposed() == b_part_shape_stride.is_transposed())
            && a_part_shape_stride.is_contiguous()
            && b_part_shape_stride.is_contiguous()
        {
            idx = i;
        } else {
            break;
        }
    }

    idx
}

fn combine_vecs(vec1: &[usize], vec2: &[usize]) -> Vec<(usize, usize)> {
    let len1 = vec1.len();
    let len2 = vec2.len();
    let max_len = len1.max(len2);

    let mut combined_vec = Vec::with_capacity(max_len);

    let mut i = 0;
    let mut j = 0;

    for _ in 0..max_len {
        let item1 = vec1[i];
        let item2 = vec2[j];

        combined_vec.push((item1, item2));

        i = (i + 1) % len1;
        j = (j + 1) % len2;
    }

    combined_vec
}

fn get_all_blas_opset_stride(
    a: ShapeStride<DimDyn>,
    b: ShapeStride<DimDyn>,
) -> Vec<(usize, usize)> {
    let max_idx = get_max_shape_idx_of_apply_blas(a, b);
    let a_stride = a.stride();
    let b_stride = b.stride();
    let a_shape = a.shape();
    let b_shape = b.shape();

    let a_len = a.shape().len();
    let b_len = b.shape().len();

    let a_part_stride = DimDyn::from(&a_stride.slice()[..a_len - max_idx]);
    let b_part_stride = DimDyn::from(&b_stride.slice()[..b_len - max_idx]);
    let a_part_shape = DimDyn::from(&a_shape.slice()[..a_len - max_idx]);
    let b_part_shape = DimDyn::from(&b_shape.slice()[..b_len - max_idx]);

    let a_indexes = copy_apply_index(a_part_shape);
    let b_indexes = copy_apply_index(b_part_shape);

    let a_stride_offset = a_indexes.into_iter().map(|index| {
        index
            .slice()
            .iter()
            .zip(a_part_stride.slice().iter())
            .map(|(i, s)| i * s)
            .sum()
    });

    let b_stride_offset = b_indexes.into_iter().map(|index| {
        index
            .slice()
            .iter()
            .zip(b_part_stride.slice().iter())
            .map(|(i, s)| i * s)
            .sum()
    });

    combine_vecs(
        &a_stride_offset.collect::<Vec<_>>(),
        &b_stride_offset.collect::<Vec<_>>(),
    )
}

fn copy<T: Num>(mut to: Matrix<ViewMutMem<T>, DimDyn>, source: Matrix<ViewMem<T>, DimDyn>) {
    if to.shape().is_empty() {
        unsafe {
            to.as_mut_ptr().write(source.as_ptr().read());
        }
        return;
    }

    let blas_opset_stride = get_all_blas_opset_stride(to.shape_stride(), source.shape_stride());
    let max_blas_apply_idx =
        get_max_shape_idx_of_apply_blas(to.shape_stride(), source.shape_stride());

    let to_shape = to.shape();
    let to_stride = to.stride();
    let source_stride = source.stride();

    let to_stride_ = *to_stride.slice()[to_stride.len() - max_blas_apply_idx..]
        .iter()
        .min()
        .unwrap();
    let source_stride_ = *source_stride.slice()[source_stride.len() - max_blas_apply_idx..]
        .iter()
        .min()
        .unwrap();

    let to_blas_num_elm_ =
        DimDyn::from(&to_shape.slice()[to_shape.len() - max_blas_apply_idx..]).num_elm();

    let to_ptr = to.as_mut_ptr();
    let source_ptr = source.as_ptr();

    for (to_offset, source_offset) in blas_opset_stride {
        let to_ptr = unsafe { to_ptr.offset(to_offset as isize) };
        let source_ptr = unsafe { source_ptr.offset(source_offset as isize) };
        <ViewMutMem<T> as Memory>::Blas::copy(
            to_blas_num_elm_,
            source_ptr,
            source_stride_,
            to_ptr as *mut _,
            to_stride_,
        );
    }
}

#[cfg(test)]
mod deep_copy {
    use super::*;
    use crate::{
        matrix::{
            IndexItem, MatrixSlice, MatrixSliceMut, OwnedMatrix, ToViewMatrix, ToViewMutMatrix,
        },
        matrix_impl::{OwnedMatrix1D, OwnedMatrix2D},
        slice,
    };

    #[test]
    fn get_all_blas_opset_stride_2d_2d() {
        let a = DimDyn::from(&[2, 3]);
        let b = DimDyn::from(&[2, 3]);
        let b_stride = DimDyn::from(&[3, 1]);
        let a_stride = DimDyn::from(&[3, 1]);
        let a_shape_stride = ShapeStride::new(a, a_stride);
        let b_shape_stride = ShapeStride::new(b, b_stride);

        let result = get_all_blas_opset_stride(a_shape_stride, b_shape_stride);
        assert_eq!(result, vec![(0, 0)]);
    }

    #[test]
    fn get_all_blas_opset_stride_2d_1d() {
        let a = DimDyn::from(&[2, 3]);
        let b = DimDyn::from(&[3]);
        let b_stride = DimDyn::from(&[1]);
        let a_stride = DimDyn::from(&[3, 1]);
        let a_shape_stride = ShapeStride::new(a, a_stride);
        let b_shape_stride = ShapeStride::new(b, b_stride);
        let result = get_all_blas_opset_stride(a_shape_stride, b_shape_stride);
        assert_eq!(result, vec![(0, 0), (3, 0)]);
    }

    #[test]
    fn get_all_blas_opset_stride_2d_2d_sliced() {
        let a = DimDyn::from(&[2, 3]);
        let b = DimDyn::from(&[2, 3]);
        let a_stride = DimDyn::from(&[9, 2]);
        let b_stride = DimDyn::from(&[15, 3]);
        let a_shape_stride = ShapeStride::new(a, a_stride);
        let b_shape_stride = ShapeStride::new(b, b_stride);
        let result = get_all_blas_opset_stride(a_shape_stride, b_shape_stride);
        assert_eq!(result, vec![(0, 0), (9, 15)]);
    }

    #[test]
    fn get_all_blas_opse_stride_3d_3d() {
        let a = DimDyn::from(&[2, 3, 4]);
        let b = DimDyn::from(&[2, 3, 4]);
        let a_stride = DimDyn::from(&[12, 4, 1]);
        let b_stride = DimDyn::from(&[12, 4, 1]);
        let a_shape_stride = ShapeStride::new(a, a_stride);
        let b_shape_stride = ShapeStride::new(b, b_stride);
        let result = get_all_blas_opset_stride(a_shape_stride, b_shape_stride);
        assert_eq!(result, vec![(0, 0)]);
    }

    #[test]
    fn get_all_blas_offset_stride_3d_2d() {
        let a = DimDyn::from(&[2, 3, 4]);
        let b = DimDyn::from(&[3, 4]);
        let a_stride = DimDyn::from(&[12, 4, 1]);
        let b_stride = DimDyn::from(&[4, 1]);
        let a_shape_stride = ShapeStride::new(a, a_stride);
        let b_shape_stride = ShapeStride::new(b, b_stride);
        let result = get_all_blas_opset_stride(a_shape_stride, b_shape_stride);
        assert_eq!(result, vec![(0, 0), (12, 0)]);
    }

    #[test]
    fn get_all_blas_offset_stride_3d_2d_sliced() {
        let a = DimDyn::from(&[2, 3, 4]);
        let b = DimDyn::from(&[3, 4]);
        let a_stride = DimDyn::from(&[36, 8, 1]);
        let b_stride = DimDyn::from(&[9, 1]);
        let a_shape_stride = ShapeStride::new(a, a_stride);
        let b_shape_stride = ShapeStride::new(b, b_stride);
        let result = get_all_blas_opset_stride(a_shape_stride, b_shape_stride);
        assert_eq!(
            result,
            vec![(0, 0), (8, 9), (16, 18), (36, 0), (44, 9), (52, 18),]
        );
    }

    #[test]
    fn get_all_blas_offset_stride_3d_2d_sliced_transposed() {
        let a = DimDyn::from(&[2, 3, 4]);
        let b = DimDyn::from(&[3, 4]);
        let a_stride = DimDyn::from(&[12, 4, 1]);
        let b_stride = DimDyn::from(&[1, 3]);
        let a_shape_stride = ShapeStride::new(a, a_stride);
        let b_shape_stride = ShapeStride::new(b, b_stride);
        let result = get_all_blas_opset_stride(a_shape_stride, b_shape_stride);
        assert_eq!(
            result,
            vec![(0, 0), (4, 1), (8, 2), (12, 0), (16, 1), (20, 2)]
        );
    }

    #[test]
    fn get_all_blas_offset_stride_4d_4d_swap_index() {
        // 元が4, 2, 3, 5で2, 0, 1, 3でトランスポーズ
        let a = DimDyn::from(&[2, 3, 4, 5]);
        let b = DimDyn::from(&[2, 3, 4, 5]);
        let a_stride = DimDyn::from(&[15, 5, 30, 1]);
        let b_stride = DimDyn::from(&[60, 20, 5, 1]);
        let result =
            get_all_blas_opset_stride(ShapeStride::new(a, a_stride), ShapeStride::new(b, b_stride));
        assert_eq!(
            result,
            vec![
                (0, 0),
                (30, 5),
                (60, 10),
                (90, 15),
                (5, 20),
                (35, 25),
                (65, 30),
                (95, 35),
                (10, 40),
                (40, 45),
                (70, 50),
                (100, 55),
                (15, 60),
                (45, 65),
                (75, 70),
                (105, 75),
                (20, 80),
                (50, 85),
                (80, 90),
                (110, 95),
                (25, 100),
                (55, 105),
                (85, 110),
                (115, 115)
            ]
        );
    }

    #[test]
    fn default_stride_1d() {
        let a = vec![0f32; 6];
        let b = vec![1f32, 2., 3., 4., 5., 6.];

        let mut a = OwnedMatrix1D::from_vec(a, [6]);
        let b = OwnedMatrix1D::from_vec(b, [6]);

        let a_view_mut = a.to_view_mut();

        a_view_mut
            .into_dyn_dim()
            .to_view_mut()
            .copy_from(&b.to_view().into_dyn_dim());

        assert_eq!(a.index_item([0]), 1.);
        assert_eq!(a.index_item([1]), 2.);
        assert_eq!(a.index_item([2]), 3.);
        assert_eq!(a.index_item([3]), 4.);
        assert_eq!(a.index_item([4]), 5.);
        assert_eq!(a.index_item([5]), 6.);
    }

    #[test]
    fn sliced_1d() {
        let a = vec![0f32; 6];
        let v = vec![0f32, 1., 2., 3., 4., 5.];

        let mut a = OwnedMatrix1D::from_vec(a.clone(), [6]);
        let v = OwnedMatrix1D::from_vec(v, [6]);

        let a_sliced = a.slice_mut(slice!(..;2));
        let v_sliced = v.slice(slice!(0..3));

        a_sliced.into_dyn_dim().copy_from(&v_sliced.into_dyn_dim());
        assert_eq!(a.index_item([0]), 0.);
        assert_eq!(a.index_item([1]), 0.);
        assert_eq!(a.index_item([2]), 1.);
        assert_eq!(a.index_item([3]), 0.);
        assert_eq!(a.index_item([4]), 2.);
        assert_eq!(a.index_item([5]), 0.);
    }

    #[test]
    fn defualt_stride_2d() {
        let a = vec![0f32; 6];
        let b = vec![1f32, 2., 3., 4., 5., 6.];

        let mut a = OwnedMatrix2D::from_vec(a, [2, 3]);
        let b = OwnedMatrix2D::from_vec(b, [2, 3]);

        let a_view_mut = a.to_view_mut();

        a_view_mut
            .into_dyn_dim()
            .to_view_mut()
            .copy_from(&b.to_view().into_dyn_dim());

        assert_eq!(a.index_item([0, 0]), 1.);
        assert_eq!(a.index_item([0, 1]), 2.);
        assert_eq!(a.index_item([0, 2]), 3.);
        assert_eq!(a.index_item([1, 0]), 4.);
        assert_eq!(a.index_item([1, 1]), 5.);
        assert_eq!(a.index_item([1, 2]), 6.);
    }

    #[test]
    fn sliced_2d() {
        let a = vec![0f32; 12];
        let v = vec![0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.];
        let mut a = OwnedMatrix2D::from_vec(a.clone(), [3, 4]);
        let v = OwnedMatrix2D::from_vec(v, [3, 4]);

        let a_sliced = a.slice_mut(slice!(0..2, 0..3));
        let v_sliced = v.slice(slice!(1..3, 1..4));

        a_sliced.into_dyn_dim().copy_from(&v_sliced.into_dyn_dim());
        assert_eq!(a.index_item([0, 0]), 5.);
        assert_eq!(a.index_item([0, 1]), 6.);
        assert_eq!(a.index_item([0, 2]), 7.);
        assert_eq!(a.index_item([0, 3]), 0.);
        assert_eq!(a.index_item([1, 0]), 9.);
        assert_eq!(a.index_item([1, 1]), 10.);
        assert_eq!(a.index_item([1, 2]), 11.);
        assert_eq!(a.index_item([2, 3]), 0.);
    }
}

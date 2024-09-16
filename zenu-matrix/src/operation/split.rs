use crate::{
    device::Device,
    dim::{DimDyn, DimTrait},
    index::index_dyn_impl::Index,
    matrix::{Matrix, Owned, Repr},
    num::Num,
};

/// Splits a matrix into multiple sub-matrices along a specified axis.
///
/// # Arguments
/// * `matrix` - The matrix to split.
/// * `axis` - The axis along which to split.
/// * `num_splits` - Number of splits (must evenly divide the size along the axis).
///
/// # Panics
/// * If the axis is out of bounds.
/// * If the size along the axis is not divisible by `num_splits`.
#[must_use]
pub fn split<T: Num, R: Repr<Item = T>, D: Device>(
    matrix: &Matrix<R, DimDyn, D>,
    axis: usize,
    num_splits: usize,
) -> Vec<Matrix<Owned<T>, DimDyn, D>> {
    let shape = matrix.shape();
    let ndim = shape.len();

    assert!(axis < ndim, "Axis out of bounds");
    assert!(
        shape[axis] % num_splits == 0,
        "Size along axis {} ({}) is not divisible by num_splits ({})",
        axis,
        shape[axis],
        num_splits
    );

    let mut output_shape = shape;
    output_shape[axis] /= num_splits;

    let splited_axis = shape[axis] / num_splits;

    let mut outputs = Vec::with_capacity(num_splits);

    for i in 0..num_splits {
        let mut output = Matrix::alloc(output_shape);

        for j in 0..splited_axis {
            let view = matrix.index_axis(Index::new(axis, i * splited_axis + j));
            output
                .to_ref_mut()
                .index_axis_mut(Index::new(axis, j))
                .copy_from(&view);
        }

        outputs.push(output);
    }

    outputs
}

#[cfg(test)]
mod split_test {
    use super::*;
    use crate::{
        device::Device,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_mat_eq_epsilon, run_mat_test};

    fn test_split_axis0<D: Device>() {
        let matrix = Matrix::<Owned<f32>, _, D>::from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]);

        let splits = split(&matrix, 0, 3);

        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].shape().slice(), [1, 2]);
        assert_eq!(splits[1].shape().slice(), [1, 2]);
        assert_eq!(splits[2].shape().slice(), [1, 2]);

        let expected = vec![
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2.], [1, 2]),
            Matrix::<Owned<f32>, _, D>::from_vec(vec![3., 4.], [1, 2]),
            Matrix::<Owned<f32>, _, D>::from_vec(vec![5., 6.], [1, 2]),
        ];

        for (split, exp) in splits.iter().zip(expected.iter()) {
            assert_mat_eq_epsilon!(split, exp, 1e-6);
        }
    }
    run_mat_test!(test_split_axis0, test_split_axis0_cpu, test_split_axis0_gpu);

    fn test_split_axis1<D: Device>() {
        let matrix = Matrix::<Owned<f32>, _, D>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);

        let splits = split(&matrix, 1, 3);

        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].shape().slice(), [2, 1]);
        assert_eq!(splits[1].shape().slice(), [2, 1]);
        assert_eq!(splits[2].shape().slice(), [2, 1]);

        let expected = vec![
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 4.], [2, 1]),
            Matrix::<Owned<f32>, _, D>::from_vec(vec![2., 5.], [2, 1]),
            Matrix::<Owned<f32>, _, D>::from_vec(vec![3., 6.], [2, 1]),
        ];

        for (split, exp) in splits.iter().zip(expected.iter()) {
            let diff = split - exp;
            assert!(diff.asum() < 1e-6);
        }
    }
    run_mat_test!(test_split_axis1, test_split_axis1_cpu, test_split_axis1_gpu);
}

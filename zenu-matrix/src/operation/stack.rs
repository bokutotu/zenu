use crate::{
    device::Device,
    dim::{DimDyn, DimTrait, LessDimTrait},
    index::index_dyn_impl::Index,
    matrix::{Matrix, Owned, Repr},
    num::Num,
};

/// Stack a sequence of matrices along a new axis.
///
/// # Arguments
/// * `matrices` - A slice of matrices to stack.
/// * `axis` - The axis along which to stack the matrices.
///
/// # Panics
/// * If the matrices do not have the same shape.
/// * If the axis is out of bounds.
pub fn stack<T: Num, R: Repr<Item = T>, S: DimTrait + LessDimTrait, D: Device>(
    matrices: &[Matrix<R, S, D>],
    axis: usize,
) -> Matrix<Owned<T>, DimDyn, D> {
    assert!(!matrices.is_empty(), "No matrices to stack");
    let first_shape = matrices[0].shape();

    for m in matrices.iter().skip(1) {
        assert_eq!(
            m.shape(),
            first_shape,
            "All matrices must have the same shape"
        );
    }

    let ndim = first_shape.len();
    assert!(
        axis <= ndim,
        "Axis out of bounds for stack: axis={axis} ndim={ndim}",
    );

    let output_shape = {
        let mut shape = first_shape;
        shape[axis] *= matrices.len();
        DimDyn::from(shape.slice())
    };

    let mut result: Matrix<Owned<T>, DimDyn, D> = Matrix::alloc(output_shape);

    for (i, m) in matrices.iter().enumerate() {
        // let index = Index::new(axis, i);
        // result.to_ref_mut().index_axis_mut(index).copy_from(m);
        for j in 0..first_shape[axis] {
            let index = Index::new(axis, i * first_shape[axis] + j);
            result
                .to_ref_mut()
                .index_axis_mut(index)
                .copy_from(&m.index_axis(Index::new(axis, j)));
        }
    }

    result
}

#[cfg(test)]
mod stack_test {
    use super::*;
    use crate::{
        device::Device,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_mat_eq_epsilon, run_mat_test};

    fn test_stack_axis0<D: Device>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3.], [3]);
        let b = Matrix::<Owned<f32>, _, D>::from_vec(vec![4., 5., 6.], [3]);
        let c = Matrix::<Owned<f32>, _, D>::from_vec(vec![7., 8., 9.], [3]);

        let result = stack(&[a, b, c], 0);

        let expected = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
            [9],
        );

        assert_eq!(result.shape().slice(), [9]);
        // let diff = result - expected;
        // assert!(diff.asum() < 1e-6);
        assert_mat_eq_epsilon!(result, expected, 1e-6);
    }
    run_mat_test!(test_stack_axis0, test_stack_axis0_cpu, test_stack_axis0_gpu);

    fn test_stack_axis1<D: Device>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 4.], [2, 2]);
        let b = Matrix::<Owned<f32>, _, D>::from_vec(vec![5., 6., 7., 8.], [2, 2]);
        let c = Matrix::<Owned<f32>, _, D>::from_vec(vec![9., 10., 11., 12.], [2, 2]);

        let result = stack(&[a, b, c], 1);

        let expected = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![1., 2., 5., 6., 9., 10., 3., 4., 7., 8., 11., 12.],
            [2, 6],
        );

        assert_eq!(result.shape().slice(), [2, 6]);
        // let diff = result - expected;
        // assert!(diff.asum() < 1e-6);
        assert_mat_eq_epsilon!(result, expected, 1e-6);
    }
    run_mat_test!(test_stack_axis1, test_stack_axis1_cpu, test_stack_axis1_gpu);
}

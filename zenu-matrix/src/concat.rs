use crate::{
    device::Device,
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{Matrix, Owned, Repr},
    num::Num,
};

pub fn concat<T: Num, R: Repr<Item = T>, S: DimTrait, D: Device>(
    matrix: &[Matrix<R, S, D>],
) -> Matrix<Owned<T>, DimDyn, D> {
    let first_shape = matrix[0].shape();
    for m in matrix.iter().skip(1) {
        if m.shape() != first_shape {
            panic!("All matrices must have the same shape");
        }
    }
    if first_shape.len() == 4 {
        panic!("Concatenation of 4D matrices is not supported");
    }

    let mut shape = DimDyn::default();
    shape.push_dim(matrix.len());
    for d in first_shape {
        shape.push_dim(d);
    }

    let mut result = Matrix::zeros(shape);

    for (i, m) in matrix.iter().enumerate() {
        let view = m.to_ref().into_dyn_dim();
        result
            .to_ref_mut()
            .index_axis_mut_dyn(Index0D::new(i))
            .copy_from(&view);
    }

    result
}

#[cfg(test)]
mod concat {
    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    fn cat_1d<D: Device>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3.], [3]);
        let b = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![4., 5., 6.], [3]);
        let c = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![7., 8., 9.], [3]);
        let result = super::concat(&[a, b, c]);

        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
            [3, 3],
        );

        let diff = result - ans;
        assert_eq!(diff.asum(), 0.);
    }
    #[test]
    fn cal_1d_cpu() {
        cat_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn cal_1d_gpu() {
        cat_1d::<crate::device::nvidia::Nvidia>();
    }

    fn cal_2d<D: Device>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 4.], [2, 2]);
        let b = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![5., 6., 7., 8.], [2, 2]);
        let c = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![9., 10., 11., 12.], [2, 2]);
        let result = super::concat(&[a, b, c]);

        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [3, 2, 2],
        );

        let diff = result - ans;
        assert_eq!(diff.asum(), 0.);
    }
    #[test]
    fn cal_2d_cpu() {
        cal_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn cal_2d_gpu() {
        cal_2d::<crate::device::nvidia::Nvidia>();
    }
}

use crate::{
    device::Device,
    dim::{default_stride, DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref, Repr},
    num::Num,
    shape_stride::ShapeStride,
};

impl<T: Num, R: Repr<Item = T>, S: DimTrait, D: Device> Matrix<R, S, D> {
    pub fn reshape<I: Into<DimDyn>>(&self, new_shape: I) -> Matrix<Ref<&T>, DimDyn, D> {
        let new_shape = new_shape.into();
        assert_eq!(
            self.shape().num_elm(),
            new_shape.num_elm(),
            "Number of elements must be the same"
        );
        assert!(
            self.shape_stride().is_default_stride(),
            r#"""
`reshape` method is not alloc new memory.
So, This matrix is not default stride, it is not allowed to use `reshape` method.
Use `reshape_new_matrix` method instead.
            """#
        );
        let new_stride = default_stride(new_shape);
        let mut result = self.to_ref().into_dyn_dim();
        result.update_shape_stride(ShapeStride::new(new_shape, new_stride));
        result
    }
}

impl<T: Num, R: Repr<Item = T>, D: Device> Matrix<R, DimDyn, D> {
    pub fn reshape_new_matrix<I: Into<DimDyn>>(&self, new_shape: I) -> Matrix<Owned<T>, DimDyn, D> {
        let new_shape = new_shape.into();
        assert_eq!(
            self.shape().num_elm(),
            new_shape.num_elm(),
            "Number of elements must be the same"
        );
        let new_stride = default_stride(new_shape);

        let mut default_stride_matrix = self.to_ref().to_default_stride();
        default_stride_matrix.update_shape_stride(ShapeStride::new(new_shape, new_stride));
        default_stride_matrix
    }
}
impl<T: Num, S: DimTrait, D: Device> Matrix<Owned<T>, S, D> {
    pub fn reshape_mut<I: Into<DimDyn>>(&mut self, new_shape: I) -> Matrix<Ref<&mut T>, DimDyn, D> {
        let new_shape = new_shape.into();
        assert_eq!(
            self.shape().num_elm(),
            new_shape.num_elm(),
            "Number of elements must be the same"
        );
        assert!(
            self.shape_stride().is_default_stride(),
            r#"""
`reshape` method is not alloc new memory.
So, This matrix is not default stride, it is not allowed to use `reshape` method.
Use `reshape_new_matrix` method instead.
            """#
        );
        let new_stride = default_stride(new_shape);
        let mut result = self.to_ref_mut().into_dyn_dim();
        result.update_shape_stride(ShapeStride::new(new_shape, new_stride));
        result
    }
}

impl<T: Num, S: DimTrait, D: Device> Matrix<Owned<T>, S, D> {
    pub fn reshape_no_alloc_owned<I: Into<DimDyn>>(
        self,
        new_shape: I,
    ) -> Matrix<Owned<T>, DimDyn, D> {
        let new_shape = new_shape.into();
        assert_eq!(
            self.shape().num_elm(),
            new_shape.num_elm(),
            "Number of elements must be the same"
        );
        assert!(
            self.shape_stride().is_default_stride(),
            r#"""
`reshape` method is not alloc new memory.
So, This matrix is not default stride, it is not allowed to use `reshape` method.
Use `reshape_new_matrix` method instead.
            """#
        );
        let mut s = self.into_dyn_dim();
        let new_shape_stride = ShapeStride::new(new_shape, default_stride(new_shape));
        s.update_shape_stride(new_shape_stride);
        s
    }
}

#[cfg(test)]
mod reshape {
    use crate::{
        device::Device,
        dim::{DimDyn, DimTrait},
        matrix::{Matrix, Owned},
    };

    fn reshape_3d_1d<D: Device>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            ],
            [2, 3, 3],
        );
        let b = a.reshape([18]);
        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            ],
            [18],
        );
        assert_eq!(b.shape().slice(), ans.shape().slice());
        assert!((b - ans).to_ref().asum() < 1e-6);
    }
    #[test]
    fn reshape_3d_1d_cpu() {
        reshape_3d_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn reshape_3d_1d_gpu() {
        reshape_3d_1d::<crate::device::nvidia::Nvidia>();
    }

    fn reshape_1d_3d<D: Device>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 4., 5., 6.], [6]);
        let b = a.reshape([2, 3, 1]);
        let ans =
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3, 1]);
        assert_eq!(b.shape().slice(), ans.shape().slice());
        assert!((b - ans).to_ref().asum() < 1e-6);
    }
    #[test]
    fn reshape_1d_3d_cpu() {
        reshape_1d_3d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn reshape_1d_3d_gpu() {
        reshape_1d_3d::<crate::device::nvidia::Nvidia>();
    }

    // #[test]
    fn reshape_new_matrix_3d_1d<D: Device>() {
        let mut a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            ],
            [2, 3, 3],
        );
        let mut a_ref_mut = a.to_ref_mut();
        a_ref_mut.transpose_by_index(&[2, 1, 0]);
        let b = a_ref_mut.reshape_new_matrix([18]);
        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![
                1., 10., 4., 13., 7., 16., 2., 11., 5., 14., 8., 17., 3., 12., 6., 15., 9., 18.,
            ],
            [18],
        );
        assert_eq!(b.shape().slice(), ans.shape().slice());
        assert!((b - ans).to_ref().asum() < 1e-6);
    }
    #[test]
    fn reshape_new_matrix_3d_1d_cpu() {
        reshape_new_matrix_3d_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn reshape_new_matrix_3d_1d_gpu() {
        reshape_new_matrix_3d_1d::<crate::device::nvidia::Nvidia>();
    }
}

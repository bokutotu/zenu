use crate::{
    device::Device,
    dim::DimTrait,
    matrix::{Matrix, Owned, Repr},
};

impl<R: Repr, S: DimTrait, D: Device> Matrix<R, S, D> {
    pub fn to_default_stride(&self) -> Matrix<Owned<R::Item>, S, D> {
        let mut output: Matrix<Owned<R::Item>, S, D> = Matrix::zeros_like(self);
        {
            let output_view_mut = output.to_ref_mut();
            output_view_mut.copy_from(self.to_ref().into_dyn_dim());
        }
        output
    }
}

#[cfg(test)]
mod to_default_stride {
    use crate::{
        dim::{default_stride, DimDyn},
        slice_dynamic,
    };

    use super::*;

    fn test_1d<D: Device>() {
        // 0 t0 16 f32 vec
        let v = vec![
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        ];

        let m: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(v.clone(), [16]);
        let sliced = m.slice(slice_dynamic!(..;2));
        // let default_strided: OwnedMatrixDyn<f32> = ToDefaultStride::to_default_stride(&sliced);
        let default_strided = sliced.to_default_stride();

        assert_eq!(
            default_strided.shape_stride().stride(),
            (default_strided.shape_stride().stride())
        );

        assert_eq!(default_strided.index_item([0]), 0.);
        assert_eq!(default_strided.index_item([1]), 2.);
        assert_eq!(default_strided.index_item([2]), 4.);
        assert_eq!(default_strided.index_item([3]), 6.);
        assert_eq!(default_strided.index_item([4]), 8.);
    }
    #[test]
    fn test_1d_cpu() {
        test_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn test_1d_gpu() {
        test_1d::<crate::device::nvidia::Nvidia>();
    }

    fn test_2d<D: Device>() {
        // 0 t0 16 f32 vec
        let v = vec![
            0., 1., 2., 3., 4., 5., 6., 7., //
            8., 9., 10., 11., 12., 13., 14., 15.,
        ];

        let m: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(v.clone(), [4, 4]);
        let sliced = m.slice(slice_dynamic!(..;2, ..;2));
        let default_strided = sliced.to_default_stride();

        assert_eq!(
            default_strided.shape_stride().stride(),
            default_stride(default_strided.shape_stride().shape())
        );

        assert_eq!(default_strided.index_item([0, 0]), 0.);
        assert_eq!(default_strided.index_item([0, 1]), 2.);
        assert_eq!(default_strided.index_item([1, 0]), 8.);
        assert_eq!(default_strided.index_item([1, 1]), 10.);
    }
    #[test]
    fn test_2d_cpu() {
        test_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn test_2d_gpu() {
        test_2d::<crate::device::nvidia::Nvidia>();
    }
}

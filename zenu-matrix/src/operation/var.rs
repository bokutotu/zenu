use crate::{
    device::Device,
    dim::DimDyn,
    matrix::{Matrix, Owned, Repr},
    num::Num,
};

// use crate::{
//     dim::DimDyn, matrix::ToViewMatrix, matrix_impl::Matrix, memory_impl::OwnedMem, num::Num,
// };
//
// use super::mean::Mean;
//
// pub trait Variance<T: Num> {
//     fn variance(&self, axis: Option<usize>, keep_dim: bool) -> Matrix<OwnedMem<T>, DimDyn>;
// }
//
impl<T: Num, R: Repr<Item = T>, D: Device> Matrix<R, DimDyn, D> {
    pub fn variance(&self, axis: Option<usize>, keep_dim: bool) -> Matrix<Owned<T>, DimDyn, D> {
        let mean = self.mean(axis, true);
        let diff = self.to_ref() - mean;
        let diff = diff.to_ref() * diff.to_ref();
        diff.mean(axis, keep_dim)
    }
}

#[cfg(test)]
mod variance {
    use crate::{device::Device, dim::DimDyn, matrix::Matrix};

    fn variance_1d<D: Device>() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let x = Matrix::<_, DimDyn, D>::from_vec(x, &[4]);
        let ans = x.variance(None, false);
        assert!((ans - 1.25).asum() < 1e-6);
    }
    #[test]
    fn variance_1d_cpu() {
        variance_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn variance_1d_cuda() {
        variance_1d::<crate::device::nvidia::Nvidia>();
    }

    fn variance_1d_<D: Device>() {
        let x = vec![1.0, 2.0];
        let x = Matrix::<_, DimDyn, D>::from_vec(x, &[2]);
        let ans = x.variance(None, false);
        assert!((ans - 0.25).asum() < 1e-6);
    }
    #[test]
    fn variance_1d_cpu_() {
        variance_1d_::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn variance_1d_cuda_() {
        variance_1d_::<crate::device::nvidia::Nvidia>();
    }

    fn variance_2d<D: Device>() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let x = Matrix::<_, DimDyn, D>::from_vec(x, &[2, 2]);
        let ans = x.variance(None, false);
        assert!((ans - 1.25).asum() < 1e-6);
    }
    #[test]
    fn variance_2d_cpu() {
        variance_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn variance_2d_cuda() {
        variance_2d::<crate::device::nvidia::Nvidia>();
    }

    fn variance_2d_0<D: Device>() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let x = Matrix::<_, DimDyn, D>::from_vec(x, [2, 2]);
        let ans = x.variance(Some(0), false);
        assert!((ans - 1.0).asum() < 1e-6);
    }
    #[test]
    fn variance_2d_0_cpu() {
        variance_2d_0::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn variance_2d_0_cuda() {
        variance_2d_0::<crate::device::nvidia::Nvidia>();
    }

    fn variance_2d_1<D: Device>() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let x = Matrix::<_, DimDyn, D>::from_vec(x, [2, 2]);
        let ans = x.variance(Some(1), false);
        assert!((ans - 0.25).asum() < 1e-6);
    }
    #[test]
    fn variance_2d_1_cpu() {
        variance_2d_1::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn variance_2d_1_cuda() {
        variance_2d_1::<crate::device::nvidia::Nvidia>();
    }
}

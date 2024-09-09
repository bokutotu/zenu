use std::any::TypeId;

use crate::{
    device::{cpu::Cpu, DeviceBase},
    dim::DimTrait,
    index::Index0D,
    matrix::{Matrix, Repr},
    num::Num,
};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

pub trait Asum: DeviceBase {
    fn asum<T: Num>(n: usize, x: *const T, incx: usize) -> T;
}

impl Asum for Cpu {
    fn asum<T: Num>(n: usize, x: *const T, incx: usize) -> T {
        use cblas::{dasum, sasum};
        extern crate openblas_src;

        let n = i32::try_from(n).unwrap();
        let incx = i32::try_from(incx).unwrap();

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let x = unsafe { std::slice::from_raw_parts(x.cast(), 1) };
            let result = unsafe { sasum(n, x, incx) };
            T::from_f32(result)
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let x = unsafe { std::slice::from_raw_parts(x.cast(), 1) };
            let result = unsafe { dasum(n, x, incx) };
            T::from_f64(result)
        } else {
            unimplemented!()
        }
    }
}

#[cfg(feature = "nvidia")]
impl Asum for Nvidia {
    fn asum<T: Num>(n: usize, x: *const T, incx: usize) -> T {
        use zenu_cuda::cublas::cublas_asum;

        cublas_asum(n, x, incx).unwrap()
    }
}

impl<T: Num, R: Repr<Item = T>, S: DimTrait, D: DeviceBase + Asum> Matrix<R, S, D> {
    pub fn asum(&self) -> T {
        let s = self.to_ref().into_dyn_dim();
        if s.shape().is_empty() {
            self.index_item(&[] as &[usize])
        } else if s.shape_stride().is_contiguous() {
            let num_elm = s.shape().num_elm();
            let num_dim = s.shape().len();
            let stride = s.stride();
            D::asum(num_elm, s.as_ptr(), stride[num_dim - 1])
        } else {
            let mut sum = T::zero();
            for i in 0..s.shape()[0] {
                let tmp = s.index_axis_dyn(Index0D::new(i));
                sum += tmp.asum();
            }
            sum
        }
    }
}

#[cfg(test)]
mod asum_test {
    #![expect(clippy::float_cmp)]

    use crate::{dim::DimDyn, matrix::Owned, slice_dynamic};

    use super::*;

    fn defualt_1d<D: DeviceBase + Asum>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0, 2.0, 3.0], [3]);
        assert_eq!(a.asum(), 6.0);
    }
    #[test]
    fn defualt_1d_cpu() {
        defualt_1d::<Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn defualt_1d_nvidia() {
        defualt_1d::<Nvidia>();
    }

    fn defualt_2d<D: DeviceBase + Asum>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert_eq!(a.asum(), 10.0);
    }
    #[test]
    fn defualt_2d_cpu() {
        defualt_2d::<Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn defualt_2d_nvidia() {
        defualt_2d::<Nvidia>();
    }

    fn sliced_2d<D: DeviceBase + Asum>() {
        let a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b = a.slice(slice_dynamic!(0..2, 0..1));
        assert_eq!(b.asum(), 4.0);
    }
    #[test]
    fn sliced_2d_cpu() {
        sliced_2d::<Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sliced_2d_nvidia() {
        sliced_2d::<Nvidia>();
    }
}

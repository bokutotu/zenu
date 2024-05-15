use std::any::TypeId;

use crate::{
    device::{cpu::Cpu, Device, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Repr},
    num::Num,
};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;
#[cfg(feature = "nvidia")]
use zenu_cuda::cublas::*;

pub trait MaxIdx: DeviceBase {
    fn max_idx<T: Num>(input: *const T, size: usize, stride: usize) -> usize;
}

impl MaxIdx for Cpu {
    fn max_idx<T: Num>(input: *const T, size: usize, stride: usize) -> usize {
        extern crate openblas_src;
        use cblas::*;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let input = input as *const f32;
            let input = unsafe { std::slice::from_raw_parts(input, size * stride) };
            unsafe { isamax(size as i32, input, stride as i32) as usize }
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let input = input as *const f64;
            let input = unsafe { std::slice::from_raw_parts(input, size * stride) };
            unsafe { idamax(size as i32, input, stride as i32) as usize }
        } else {
            panic!("Unsupported type");
        }
    }
}

#[cfg(feature = "nvidia")]
impl MaxIdx for Nvidia {
    fn max_idx<T: Num>(input: *const T, size: usize, stride: usize) -> usize {
        cublas_amax(size, input, stride)
            .unwrap()
            .try_into()
            .unwrap()
    }
}

impl<T: Num, R: Repr<Item = T>, D: Device> Matrix<R, DimDyn, D> {
    pub fn max_idx(&self) -> DimDyn {
        let default_stride = self.to_default_stride();
        let idx = <D as MaxIdx>::max_idx(
            default_stride.as_ptr(),
            default_stride.shape().num_elm(),
            default_stride.stride()[default_stride.shape().len() - 1],
        );
        default_stride.shape_stride().get_dim_by_offset(idx)
    }

    pub fn max_item(&self) -> T {
        let idx = self.max_idx();
        self.index_item(idx)
    }
}

#[cfg(test)]
mod max_idx {
    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
        slice_dynamic,
    };

    fn default_1d<D: Device>() {
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![0., 1., 2., 3.], [4]);
        assert_eq!(a.to_ref().max_idx(), [3].into());
    }
    #[test]
    fn default_1d_cpu() {
        default_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn default_1d_gpu() {
        default_1d::<crate::device::nvidia::Nvidia>();
    }

    fn default_2d<D: Device>() {
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![0., 1., 2., 3.], [2, 2]);
        assert_eq!(a.to_ref().max_idx(), [1, 1].into());
    }
    #[test]
    fn default_2d_cpu() {
        default_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn default_2d_gpu() {
        default_2d::<crate::device::nvidia::Nvidia>();
    }

    fn sliced_3d<D: Device>() {
        let mut v = Vec::new();
        for i in 0..8 * 8 * 8 {
            v.push(i as f32);
        }
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(v, [8, 8, 8]);
        let sliced = a.slice(slice_dynamic!(..;3, ..;4, ..;2));
        assert_eq!(sliced.max_idx(), [2, 1, 3].into());
    }
    #[test]
    fn sliced_3d_cpu() {
        sliced_3d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sliced_3d_gpu() {
        sliced_3d::<crate::device::nvidia::Nvidia>();
    }
}

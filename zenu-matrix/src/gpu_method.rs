use std::any::TypeId;

use crate::device::{cpu::Cpu, nvidia::Nvidia, DeviceBase};
use crate::dim::DimTrait;
use crate::matrix::{Matrix, Owned, Ptr};
use crate::num::Num;

impl<T, S, D> Matrix<Owned<T>, S, D>
where
    T: Num,
    D: DeviceBase,
    S: DimTrait,
{
    pub fn move_device<D2>(&self) -> Matrix<Owned<T>, S, D2>
    where
        D: 'static,
        D2: DeviceBase + 'static,
    {
        let self_raw_ptr = self.as_ptr() as *mut T;
        let len = self.shape().num_elm();

        let ptr = match (TypeId::of::<D>(), TypeId::of::<D2>()) {
            (a, b) if a == b => self_raw_ptr,
            (a, b) if a == TypeId::of::<Cpu>() && b == TypeId::of::<Nvidia>() => {
                let ptr = zenu_cuda::runtime::copy_to_gpu(self_raw_ptr, len);
                ptr
            }
            (a, b) if a == TypeId::of::<Nvidia>() && b == TypeId::of::<Cpu>() => {
                let ptr = zenu_cuda::runtime::copy_to_cpu(self_raw_ptr, len);
                ptr
            }
            _ => unreachable!(),
        };

        let ptr = Ptr::new(ptr, len, self.offset());

        Matrix::new(ptr, self.shape(), self.stride())
    }
}

#[cfg(test)]
mod nvidia {
    use crate::dim::DimDyn;

    use super::{Cpu, Matrix, Nvidia, Owned};

    #[test]
    fn move_cpu_gpu_cpu() {
        let vec = vec![1.0, 2.0, 3.0, 4.0];
        let cpu_mat: Matrix<Owned<f64>, DimDyn, Cpu> = Matrix::from_vec(vec, [4]);
        let gpu_mat = cpu_mat.move_device::<Nvidia>();
        let cpu_mat_2 = gpu_mat.move_device::<Cpu>();
        let cpu_mat_2_slice = cpu_mat_2.as_slice();
        assert_eq!(cpu_mat_2_slice, &[1.0, 2.0, 3.0, 4.0]);
    }
}

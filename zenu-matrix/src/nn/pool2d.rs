use crate::{
    device::{cpu::Cpu, Device},
    dim::DimDyn,
    matrix::{Matrix, Ref},
    num::Num,
};

#[cfg(feature = "nvidia")]
use zenu_cuda::cudnn::pooling::Pool2d;

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

use super::im2col::{im2col, Im2ColRes};

pub struct Pool2dConfig<T: Num> {
    #[cfg(feature = "nvidia")]
    pub config: Pool2d<T>,
    _phantom: std::marker::PhantomData<T>,
}

pub trait Pool2dImpl: Device {
    fn pool2d<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        output: Matrix<Ref<&mut T>, DimDyn, Self>,
        kernel_shape: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        config: &Pool2dConfig<T>,
    ) -> Result<(), String>;

    fn pool2d_backward<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        input_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        output: Matrix<Ref<&T>, DimDyn, Self>,
        output_grad: Matrix<Ref<&T>, DimDyn, Self>,
        kernel_shape: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        config: &Pool2dConfig<T>,
    );
}

impl Pool2dImpl for Cpu {
    fn pool2d<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        output: Matrix<Ref<&mut T>, DimDyn, Self>,
        kernel_shape: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        config: &Pool2dConfig<T>,
    ) -> Result<(), String> {
        let Im2ColRes { mut col, .. } = im2col(input, kernel_shape, stride, padding, false);
        let col_shape = col.shape();
        let col = col.reshape_no_alloc_owned([
            col_shape[0],
            col_shape[1],
            col_shape[2] * col_shape[3],
            col_shape[4],
            col_shape[5],
        ]);
        todo!();
    }

    fn pool2d_backward<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        input_grad: Matrix<Ref<&mut T>, DimDyn, Self>,
        output: Matrix<Ref<&T>, DimDyn, Self>,
        output_grad: Matrix<Ref<&T>, DimDyn, Self>,
        kernel_shape: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        config: &Pool2dConfig<T>,
    ) {
        todo!();
    }
}

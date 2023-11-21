use std::ops::{Index, IndexMut};

use crate::{
    dim::{cal_offset, Dim, DimTrait},
    memory::{CpuMemory, Memory},
};

pub trait Matrix<IT>: Index<IT> + IndexMut<IT>
where
    IT: DimTrait,
{
    type Shape: DimTrait;
    type Stride: DimTrait;
    type Memory: Memory;

    fn shape(&self) -> &Self::Shape;
    fn stride(&self) -> &Self::Stride;
    fn shape_stride(&self) -> (Self::Shape, Self::Stride);
}

pub struct CpuMatrix<T> {
    shape: Dim,
    stride: Dim,
    memory: CpuMemory<T>,
}

impl<T> CpuMatrix<T> {
    pub fn new(shape: Dim, stride: Dim, memory: CpuMemory<T>) -> Self {
        if shape.len() != stride.len() {
            panic!("shape and stride must have the same length");
        }
        Self {
            shape,
            stride,
            memory,
        }
    }

    pub fn shape(&self) -> &Dim {
        &self.shape
    }

    pub fn stride(&self) -> &Dim {
        &self.stride
    }
}

impl<I, T> Index<I> for CpuMatrix<T>
where
    I: DimTrait,
{
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        let offset = cal_offset(index, self.stride().clone());
        &self.memory[offset]
    }
}

impl<I, T> IndexMut<I> for CpuMatrix<T>
where
    I: DimTrait,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let offset = cal_offset(index, self.stride().clone());
        &mut self.memory[offset]
    }
}

impl<I, T: Default + Copy> Matrix<I> for CpuMatrix<T>
where
    I: DimTrait,
{
    type Shape = Dim;
    type Stride = Dim;
    type Memory = CpuMemory<T>;

    fn shape(&self) -> &Self::Shape {
        &self.shape
    }

    fn stride(&self) -> &Self::Stride {
        &self.stride
    }

    fn shape_stride(&self) -> (Dim, Dim) {
        (self.shape().clone(), self.stride().clone())
    }
}

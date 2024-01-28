use std::fmt::Debug;

use ruml_matrix::matrix_impl::{
    CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix3D, CpuOwnedMatrix4D,
};

pub trait Zero {
    fn zero(dim: &[usize]) -> Self;
}

pub trait One {
    fn one(dim: &[usize]) -> Self;
}

pub trait Value:
    Zero + One + Clone + Debug + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> + 'static
{
}

impl Zero for f32 {
    fn zero(dim: &[usize]) -> Self {
        if !dim.is_empty() {
            panic!("dim must be empty")
        }
        0.0
    }
}

impl One for f32 {
    fn one(dim: &[usize]) -> Self {
        if !dim.is_empty() {
            panic!("dim must be empty")
        }
        1.0
    }
}

impl Zero for f64 {
    fn zero(dim: &[usize]) -> Self {
        if !dim.is_empty() {
            panic!("dim must be empty")
        }
        0.0
    }
}

impl One for f64 {
    fn one(dim: &[usize]) -> Self {
        if !dim.is_empty() {
            panic!("dim must be empty")
        }
        1.0
    }
}

impl Value for f32 {}

impl Value for f64 {}

pub enum Val {
    F32(f32),
    F64(f64),
    Cpu1DF32(CpuOwnedMatrix1D<f32>),
    Cpu1DF64(CpuOwnedMatrix1D<f64>),
    Cpu2DF32(CpuOwnedMatrix2D<f32>),
    Cpu2DF64(CpuOwnedMatrix2D<f64>),
    Cpu3DF32(CpuOwnedMatrix3D<f32>),
    Cpu3DF64(CpuOwnedMatrix3D<f64>),
    Cpu4DF32(CpuOwnedMatrix4D<f32>),
    Cpu4DF64(CpuOwnedMatrix4D<f64>),
}

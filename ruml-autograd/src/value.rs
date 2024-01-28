use std::{
    fmt::Debug,
    ops::{Add, Mul},
};

use ruml_matrix::{
    matrix::{MatrixBase, ToViewMatrix},
    matrix_impl::{CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix3D, CpuOwnedMatrix4D},
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

#[derive(Clone)]
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

macro_rules! impl_is_method {
    ($method_name:ident, $type:ident) => {
        pub fn $method_name(&self) -> bool {
            matches!(self, Val::$type(_))
        }
    };
}

impl Val {
    pub fn dim(&self) -> Vec<usize> {
        match self {
            Val::F32(_) => vec![],
            Val::F64(_) => vec![],
            Val::Cpu1DF32(v) => v.shape().into_iter().collect(),
            Val::Cpu1DF64(v) => v.shape().into_iter().collect(),
            Val::Cpu2DF32(v) => v.shape().into_iter().collect(),
            Val::Cpu2DF64(v) => v.shape().into_iter().collect(),
            Val::Cpu3DF32(v) => v.shape().into_iter().collect(),
            Val::Cpu3DF64(v) => v.shape().into_iter().collect(),
            Val::Cpu4DF32(v) => v.shape().into_iter().collect(),
            Val::Cpu4DF64(v) => v.shape().into_iter().collect(),
        }
    }

    pub fn num_dim(&self) -> usize {
        self.dim().len()
    }

    impl_is_method!(is_f32, F32);
    impl_is_method!(is_f64, F64);
    impl_is_method!(is_cpu1df32, Cpu1DF32);
    impl_is_method!(is_cpu1df64, Cpu1DF64);
    impl_is_method!(is_cpu2df32, Cpu2DF32);
    impl_is_method!(is_cpu2df64, Cpu2DF64);
    impl_is_method!(is_cpu3df32, Cpu3DF32);
    impl_is_method!(is_cpu3df64, Cpu3DF64);
    impl_is_method!(is_cpu4df32, Cpu4DF32);
    impl_is_method!(is_cpu4df64, Cpu4DF64);
}

impl From<f32> for Val {
    fn from(v: f32) -> Self {
        Val::F32(v)
    }
}

impl From<f64> for Val {
    fn from(v: f64) -> Self {
        Val::F64(v)
    }
}

impl From<CpuOwnedMatrix1D<f32>> for Val {
    fn from(v: CpuOwnedMatrix1D<f32>) -> Self {
        Val::Cpu1DF32(v)
    }
}

impl From<CpuOwnedMatrix1D<f64>> for Val {
    fn from(v: CpuOwnedMatrix1D<f64>) -> Self {
        Val::Cpu1DF64(v)
    }
}

impl From<CpuOwnedMatrix2D<f32>> for Val {
    fn from(v: CpuOwnedMatrix2D<f32>) -> Self {
        Val::Cpu2DF32(v)
    }
}

impl From<CpuOwnedMatrix2D<f64>> for Val {
    fn from(v: CpuOwnedMatrix2D<f64>) -> Self {
        Val::Cpu2DF64(v)
    }
}

impl From<CpuOwnedMatrix3D<f32>> for Val {
    fn from(v: CpuOwnedMatrix3D<f32>) -> Self {
        Val::Cpu3DF32(v)
    }
}

impl From<CpuOwnedMatrix3D<f64>> for Val {
    fn from(v: CpuOwnedMatrix3D<f64>) -> Self {
        Val::Cpu3DF64(v)
    }
}

impl From<CpuOwnedMatrix4D<f32>> for Val {
    fn from(v: CpuOwnedMatrix4D<f32>) -> Self {
        Val::Cpu4DF32(v)
    }
}

impl From<CpuOwnedMatrix4D<f64>> for Val {
    fn from(v: CpuOwnedMatrix4D<f64>) -> Self {
        Val::Cpu4DF64(v)
    }
}

impl TryFrom<Val> for f32 {
    type Error = &'static str;
    fn try_from(v: Val) -> Result<Self, Self::Error> {
        match v {
            Val::F32(v) => Ok(v),
            _ => Err("type mismatch"),
        }
    }
}

impl TryFrom<Val> for f64 {
    type Error = &'static str;
    fn try_from(v: Val) -> Result<Self, Self::Error> {
        match v {
            Val::F64(v) => Ok(v),
            _ => Err("type mismatch"),
        }
    }
}

impl TryFrom<Val> for CpuOwnedMatrix1D<f32> {
    type Error = &'static str;
    fn try_from(v: Val) -> Result<Self, Self::Error> {
        match v {
            Val::Cpu1DF32(v) => Ok(v),
            _ => Err("type mismatch"),
        }
    }
}

impl TryFrom<Val> for CpuOwnedMatrix1D<f64> {
    type Error = &'static str;
    fn try_from(v: Val) -> Result<Self, Self::Error> {
        match v {
            Val::Cpu1DF64(v) => Ok(v),
            _ => Err("type mismatch"),
        }
    }
}

impl TryFrom<Val> for CpuOwnedMatrix2D<f32> {
    type Error = &'static str;
    fn try_from(v: Val) -> Result<Self, Self::Error> {
        match v {
            Val::Cpu2DF32(v) => Ok(v),
            _ => Err("type mismatch"),
        }
    }
}

impl TryFrom<Val> for CpuOwnedMatrix2D<f64> {
    type Error = &'static str;
    fn try_from(v: Val) -> Result<Self, Self::Error> {
        match v {
            Val::Cpu2DF64(v) => Ok(v),
            _ => Err("type mismatch"),
        }
    }
}

impl TryFrom<Val> for CpuOwnedMatrix3D<f32> {
    type Error = &'static str;
    fn try_from(v: Val) -> Result<Self, Self::Error> {
        match v {
            Val::Cpu3DF32(v) => Ok(v),
            _ => Err("type mismatch"),
        }
    }
}

impl TryFrom<Val> for CpuOwnedMatrix3D<f64> {
    type Error = &'static str;
    fn try_from(v: Val) -> Result<Self, Self::Error> {
        match v {
            Val::Cpu3DF64(v) => Ok(v),
            _ => Err("type mismatch"),
        }
    }
}

impl TryFrom<Val> for CpuOwnedMatrix4D<f32> {
    type Error = &'static str;
    fn try_from(v: Val) -> Result<Self, Self::Error> {
        match v {
            Val::Cpu4DF32(v) => Ok(v),
            _ => Err("type mismatch"),
        }
    }
}

impl TryFrom<Val> for CpuOwnedMatrix4D<f64> {
    type Error = &'static str;
    fn try_from(v: Val) -> Result<Self, Self::Error> {
        match v {
            Val::Cpu4DF64(v) => Ok(v),
            _ => Err("type mismatch"),
        }
    }
}

impl Add<Val> for Val {
    type Output = Val;
    fn add(self, rhs: Val) -> Self::Output {
        match (self, rhs) {
            (Val::F32(a), Val::F32(b)) => Val::F32(a + b),
            (Val::F64(a), Val::F64(b)) => Val::F64(a + b),

            (Val::Cpu1DF32(a), Val::Cpu1DF32(b)) => Val::Cpu1DF32(a.to_view() + b.to_view()),
            (Val::Cpu2DF32(a), Val::Cpu2DF32(b)) => Val::Cpu2DF32(a.to_view() + b.to_view()),
            (Val::Cpu3DF32(a), Val::Cpu3DF32(b)) => Val::Cpu3DF32(a.to_view() + b.to_view()),
            (Val::Cpu4DF32(a), Val::Cpu4DF32(b)) => Val::Cpu4DF32(a.to_view() + b.to_view()),

            (Val::Cpu1DF32(a), Val::F32(b)) => Val::Cpu1DF32(a.to_view() + b),
            (Val::Cpu2DF32(a), Val::F32(b)) => Val::Cpu2DF32(a.to_view() + b),
            (Val::Cpu3DF32(a), Val::F32(b)) => Val::Cpu3DF32(a.to_view() + b),
            (Val::Cpu4DF32(a), Val::F32(b)) => Val::Cpu4DF32(a.to_view() + b),

            (Val::Cpu2DF32(a), Val::Cpu1DF32(b)) | (Val::Cpu1DF32(b), Val::Cpu2DF32(a)) => {
                Val::Cpu2DF32(a.to_view() + b.to_view())
            }
            (Val::Cpu3DF32(a), Val::Cpu1DF32(b)) | (Val::Cpu1DF32(b), Val::Cpu3DF32(a)) => {
                Val::Cpu3DF32(a.to_view() + b.to_view())
            }
            (Val::Cpu4DF32(a), Val::Cpu1DF32(b)) | (Val::Cpu1DF32(b), Val::Cpu4DF32(a)) => {
                Val::Cpu4DF32(a.to_view() + b.to_view())
            }
            (Val::Cpu3DF32(a), Val::Cpu2DF32(b)) | (Val::Cpu2DF32(b), Val::Cpu3DF32(a)) => {
                Val::Cpu3DF32(a.to_view() + b.to_view())
            }
            (Val::Cpu4DF32(a), Val::Cpu2DF32(b)) | (Val::Cpu2DF32(b), Val::Cpu4DF32(a)) => {
                Val::Cpu4DF32(a.to_view() + b.to_view())
            }
            (Val::Cpu4DF32(a), Val::Cpu3DF32(b)) | (Val::Cpu3DF32(b), Val::Cpu4DF32(a)) => {
                Val::Cpu4DF32(a.to_view() + b.to_view())
            }

            (Val::Cpu1DF64(a), Val::Cpu1DF64(b)) => Val::Cpu1DF64(a.to_view() + b.to_view()),
            (Val::Cpu2DF64(a), Val::Cpu2DF64(b)) => Val::Cpu2DF64(a.to_view() + b.to_view()),
            (Val::Cpu3DF64(a), Val::Cpu3DF64(b)) => Val::Cpu3DF64(a.to_view() + b.to_view()),
            (Val::Cpu4DF64(a), Val::Cpu4DF64(b)) => Val::Cpu4DF64(a.to_view() + b.to_view()),

            (Val::Cpu1DF64(a), Val::F64(b)) => Val::Cpu1DF64(a.to_view() + b),
            (Val::Cpu2DF64(a), Val::F64(b)) => Val::Cpu2DF64(a.to_view() + b),
            (Val::Cpu3DF64(a), Val::F64(b)) => Val::Cpu3DF64(a.to_view() + b),
            (Val::Cpu4DF64(a), Val::F64(b)) => Val::Cpu4DF64(a.to_view() + b),

            (Val::Cpu2DF64(a), Val::Cpu1DF64(b)) | (Val::Cpu1DF64(b), Val::Cpu2DF64(a)) => {
                Val::Cpu2DF64(a.to_view() + b.to_view())
            }
            (Val::Cpu3DF64(a), Val::Cpu1DF64(b)) | (Val::Cpu1DF64(b), Val::Cpu3DF64(a)) => {
                Val::Cpu3DF64(a.to_view() + b.to_view())
            }
            (Val::Cpu4DF64(a), Val::Cpu1DF64(b)) | (Val::Cpu1DF64(b), Val::Cpu4DF64(a)) => {
                Val::Cpu4DF64(a.to_view() + b.to_view())
            }
            (Val::Cpu3DF64(a), Val::Cpu2DF64(b)) | (Val::Cpu2DF64(b), Val::Cpu3DF64(a)) => {
                Val::Cpu3DF64(a.to_view() + b.to_view())
            }
            (Val::Cpu4DF64(a), Val::Cpu2DF64(b)) | (Val::Cpu2DF64(b), Val::Cpu4DF64(a)) => {
                Val::Cpu4DF64(a.to_view() + b.to_view())
            }
            (Val::Cpu4DF64(a), Val::Cpu3DF64(b)) | (Val::Cpu3DF64(b), Val::Cpu4DF64(a)) => {
                Val::Cpu4DF64(a.to_view() + b.to_view())
            }
            _ => panic!("type mismatch"),
        }
    }
}

impl Mul<Val> for Val {
    type Output = Val;
    fn mul(self, rhs: Val) -> Self::Output {
        match (self, rhs) {
            (Val::F32(a), Val::F32(b)) => Val::F32(a * b),
            (Val::F64(a), Val::F64(b)) => Val::F64(a * b),

            (Val::Cpu1DF32(a), Val::Cpu1DF32(b)) => Val::Cpu1DF32(a.to_view() * b.to_view()),
            (Val::Cpu2DF32(a), Val::Cpu2DF32(b)) => Val::Cpu2DF32(a.to_view() * b.to_view()),
            (Val::Cpu3DF32(a), Val::Cpu3DF32(b)) => Val::Cpu3DF32(a.to_view() * b.to_view()),
            (Val::Cpu4DF32(a), Val::Cpu4DF32(b)) => Val::Cpu4DF32(a.to_view() * b.to_view()),

            (Val::Cpu1DF32(a), Val::F32(b)) => Val::Cpu1DF32(a.to_view() * b),
            (Val::Cpu2DF32(a), Val::F32(b)) => Val::Cpu2DF32(a.to_view() * b),
            (Val::Cpu3DF32(a), Val::F32(b)) => Val::Cpu3DF32(a.to_view() * b),
            (Val::Cpu4DF32(a), Val::F32(b)) => Val::Cpu4DF32(a.to_view() * b),

            (Val::Cpu2DF32(a), Val::Cpu1DF32(b)) | (Val::Cpu1DF32(b), Val::Cpu2DF32(a)) => {
                Val::Cpu2DF32(a.to_view() * b.to_view())
            }
            (Val::Cpu3DF32(a), Val::Cpu1DF32(b)) | (Val::Cpu1DF32(b), Val::Cpu3DF32(a)) => {
                Val::Cpu3DF32(a.to_view() * b.to_view())
            }
            (Val::Cpu4DF32(a), Val::Cpu1DF32(b)) | (Val::Cpu1DF32(b), Val::Cpu4DF32(a)) => {
                Val::Cpu4DF32(a.to_view() * b.to_view())
            }
            (Val::Cpu3DF32(a), Val::Cpu2DF32(b)) | (Val::Cpu2DF32(b), Val::Cpu3DF32(a)) => {
                Val::Cpu3DF32(a.to_view() * b.to_view())
            }
            (Val::Cpu4DF32(a), Val::Cpu2DF32(b)) | (Val::Cpu2DF32(b), Val::Cpu4DF32(a)) => {
                Val::Cpu4DF32(a.to_view() * b.to_view())
            }
            (Val::Cpu4DF32(a), Val::Cpu3DF32(b)) | (Val::Cpu3DF32(b), Val::Cpu4DF32(a)) => {
                Val::Cpu4DF32(a.to_view() * b.to_view())
            }

            (Val::Cpu1DF64(a), Val::Cpu1DF64(b)) => Val::Cpu1DF64(a.to_view() * b.to_view()),
            (Val::Cpu2DF64(a), Val::Cpu2DF64(b)) => Val::Cpu2DF64(a.to_view() * b.to_view()),
            (Val::Cpu3DF64(a), Val::Cpu3DF64(b)) => Val::Cpu3DF64(a.to_view() * b.to_view()),
            (Val::Cpu4DF64(a), Val::Cpu4DF64(b)) => Val::Cpu4DF64(a.to_view() * b.to_view()),

            (Val::Cpu1DF64(a), Val::F64(b)) => Val::Cpu1DF64(a.to_view() * b),
            (Val::Cpu2DF64(a), Val::F64(b)) => Val::Cpu2DF64(a.to_view() * b),
            (Val::Cpu3DF64(a), Val::F64(b)) => Val::Cpu3DF64(a.to_view() * b),
            (Val::Cpu4DF64(a), Val::F64(b)) => Val::Cpu4DF64(a.to_view() * b),

            (Val::Cpu2DF64(a), Val::Cpu1DF64(b)) | (Val::Cpu1DF64(b), Val::Cpu2DF64(a)) => {
                Val::Cpu2DF64(a.to_view() * b.to_view())
            }
            (Val::Cpu3DF64(a), Val::Cpu1DF64(b)) | (Val::Cpu1DF64(b), Val::Cpu3DF64(a)) => {
                Val::Cpu3DF64(a.to_view() * b.to_view())
            }
            (Val::Cpu4DF64(a), Val::Cpu1DF64(b)) | (Val::Cpu1DF64(b), Val::Cpu4DF64(a)) => {
                Val::Cpu4DF64(a.to_view() * b.to_view())
            }
            (Val::Cpu3DF64(a), Val::Cpu2DF64(b)) | (Val::Cpu2DF64(b), Val::Cpu3DF64(a)) => {
                Val::Cpu3DF64(a.to_view() * b.to_view())
            }
            (Val::Cpu4DF64(a), Val::Cpu2DF64(b)) | (Val::Cpu2DF64(b), Val::Cpu4DF64(a)) => {
                Val::Cpu4DF64(a.to_view() * b.to_view())
            }
            (Val::Cpu4DF64(a), Val::Cpu3DF64(b)) | (Val::Cpu3DF64(b), Val::Cpu4DF64(a)) => {
                Val::Cpu4DF64(a.to_view() * b.to_view())
            }
            _ => panic!("type mismatch"),
        }
    }
}

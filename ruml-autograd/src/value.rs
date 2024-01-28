use std::{
    fmt::Debug,
    ops::{Add, Mul},
};

use ruml_matrix::{
    matrix::ToViewMatrix,
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

impl Val {
    pub fn from_f32(v: f32) -> Self {
        Val::F32(v)
    }

    pub fn from_f64(v: f64) -> Self {
        Val::F64(v)
    }

    pub fn from_cpu1d_f32(v: CpuOwnedMatrix1D<f32>) -> Self {
        Val::Cpu1DF32(v)
    }

    pub fn from_cpu1d_f64(v: CpuOwnedMatrix1D<f64>) -> Self {
        Val::Cpu1DF64(v)
    }

    pub fn from_cpu2d_f32(v: CpuOwnedMatrix2D<f32>) -> Self {
        Val::Cpu2DF32(v)
    }

    pub fn from_cpu2d_f64(v: CpuOwnedMatrix2D<f64>) -> Self {
        Val::Cpu2DF64(v)
    }

    pub fn from_cpu3d_f32(v: CpuOwnedMatrix3D<f32>) -> Self {
        Val::Cpu3DF32(v)
    }

    pub fn from_cpu3d_f64(v: CpuOwnedMatrix3D<f64>) -> Self {
        Val::Cpu3DF64(v)
    }

    pub fn from_cpu4d_f32(v: CpuOwnedMatrix4D<f32>) -> Self {
        Val::Cpu4DF32(v)
    }

    pub fn from_cpu4d_f64(v: CpuOwnedMatrix4D<f64>) -> Self {
        Val::Cpu4DF64(v)
    }

    pub fn is_f32(&self) -> bool {
        match self {
            Val::F32(_) => true,
            _ => false,
        }
    }

    pub fn is_f64(&self) -> bool {
        match self {
            Val::F64(_) => true,
            _ => false,
        }
    }

    pub fn is_cpu1d_f32(&self) -> bool {
        match self {
            Val::Cpu1DF32(_) => true,
            _ => false,
        }
    }

    pub fn is_cpu1d_f64(&self) -> bool {
        match self {
            Val::Cpu1DF64(_) => true,
            _ => false,
        }
    }

    pub fn is_cpu2d_f32(&self) -> bool {
        match self {
            Val::Cpu2DF32(_) => true,
            _ => false,
        }
    }

    pub fn is_cpu2d_f64(&self) -> bool {
        match self {
            Val::Cpu2DF64(_) => true,
            _ => false,
        }
    }

    pub fn is_cpu3d_f32(&self) -> bool {
        match self {
            Val::Cpu3DF32(_) => true,
            _ => false,
        }
    }

    pub fn is_cpu3d_f64(&self) -> bool {
        match self {
            Val::Cpu3DF64(_) => true,
            _ => false,
        }
    }

    pub fn is_cpu4d_f32(&self) -> bool {
        match self {
            Val::Cpu4DF32(_) => true,
            _ => false,
        }
    }

    pub fn is_cpu4d_f64(&self) -> bool {
        match self {
            Val::Cpu4DF64(_) => true,
            _ => false,
        }
    }

    pub fn to_f32(&self) -> f32 {
        match self {
            Val::F32(v) => *v,
            _ => panic!("type mismatch"),
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            Val::F64(v) => *v,
            _ => panic!("type mismatch"),
        }
    }

    pub fn to_cpu1d_f32(&self) -> CpuOwnedMatrix1D<f32> {
        match self {
            Val::Cpu1DF32(v) => v.clone(),
            _ => panic!("type mismatch"),
        }
    }

    pub fn to_cpu1d_f64(&self) -> CpuOwnedMatrix1D<f64> {
        match self {
            Val::Cpu1DF64(v) => v.clone(),
            _ => panic!("type mismatch"),
        }
    }

    pub fn to_cpu2d_f32(&self) -> CpuOwnedMatrix2D<f32> {
        match self {
            Val::Cpu2DF32(v) => v.clone(),
            _ => panic!("type mismatch"),
        }
    }

    pub fn to_cpu2d_f64(&self) -> CpuOwnedMatrix2D<f64> {
        match self {
            Val::Cpu2DF64(v) => v.clone(),
            _ => panic!("type mismatch"),
        }
    }

    pub fn to_cpu3d_f32(&self) -> CpuOwnedMatrix3D<f32> {
        match self {
            Val::Cpu3DF32(v) => v.clone(),
            _ => panic!("type mismatch"),
        }
    }

    pub fn to_cpu3d_f64(&self) -> CpuOwnedMatrix3D<f64> {
        match self {
            Val::Cpu3DF64(v) => v.clone(),
            _ => panic!("type mismatch"),
        }
    }

    pub fn to_cpu4d_f32(&self) -> CpuOwnedMatrix4D<f32> {
        match self {
            Val::Cpu4DF32(v) => v.clone(),
            _ => panic!("type mismatch"),
        }
    }

    pub fn to_cpu4d_f64(&self) -> CpuOwnedMatrix4D<f64> {
        match self {
            Val::Cpu4DF64(v) => v.clone(),
            _ => panic!("type mismatch"),
        }
    }
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

            (Val::Cpu2DF32(a), Val::Cpu1DF32(b)) => Val::Cpu2DF32(a.to_view() + b.to_view()),
            (Val::Cpu3DF32(a), Val::Cpu1DF32(b)) => Val::Cpu3DF32(a.to_view() + b.to_view()),
            (Val::Cpu4DF32(a), Val::Cpu1DF32(b)) => Val::Cpu4DF32(a.to_view() + b.to_view()),
            (Val::Cpu3DF32(a), Val::Cpu2DF32(b)) => Val::Cpu3DF32(a.to_view() + b.to_view()),
            (Val::Cpu4DF32(a), Val::Cpu2DF32(b)) => Val::Cpu4DF32(a.to_view() + b.to_view()),
            (Val::Cpu4DF32(a), Val::Cpu3DF32(b)) => Val::Cpu4DF32(a.to_view() + b.to_view()),

            (Val::Cpu1DF64(a), Val::Cpu1DF64(b)) => Val::Cpu1DF64(a.to_view() + b.to_view()),
            (Val::Cpu2DF64(a), Val::Cpu2DF64(b)) => Val::Cpu2DF64(a.to_view() + b.to_view()),
            (Val::Cpu3DF64(a), Val::Cpu3DF64(b)) => Val::Cpu3DF64(a.to_view() + b.to_view()),
            (Val::Cpu4DF64(a), Val::Cpu4DF64(b)) => Val::Cpu4DF64(a.to_view() + b.to_view()),

            (Val::Cpu1DF64(a), Val::F64(b)) => Val::Cpu1DF64(a.to_view() + b),
            (Val::Cpu2DF64(a), Val::F64(b)) => Val::Cpu2DF64(a.to_view() + b),
            (Val::Cpu3DF64(a), Val::F64(b)) => Val::Cpu3DF64(a.to_view() + b),
            (Val::Cpu4DF64(a), Val::F64(b)) => Val::Cpu4DF64(a.to_view() + b),

            (Val::Cpu2DF64(a), Val::Cpu1DF64(b)) => Val::Cpu2DF64(a.to_view() + b.to_view()),
            (Val::Cpu3DF64(a), Val::Cpu1DF64(b)) => Val::Cpu3DF64(a.to_view() + b.to_view()),
            (Val::Cpu4DF64(a), Val::Cpu1DF64(b)) => Val::Cpu4DF64(a.to_view() + b.to_view()),
            (Val::Cpu3DF64(a), Val::Cpu2DF64(b)) => Val::Cpu3DF64(a.to_view() + b.to_view()),
            (Val::Cpu4DF64(a), Val::Cpu2DF64(b)) => Val::Cpu4DF64(a.to_view() + b.to_view()),
            (Val::Cpu4DF64(a), Val::Cpu3DF64(b)) => Val::Cpu4DF64(a.to_view() + b.to_view()),
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

            (Val::Cpu2DF32(a), Val::Cpu1DF32(b)) => Val::Cpu2DF32(a.to_view() * b.to_view()),
            (Val::Cpu3DF32(a), Val::Cpu1DF32(b)) => Val::Cpu3DF32(a.to_view() * b.to_view()),
            (Val::Cpu4DF32(a), Val::Cpu1DF32(b)) => Val::Cpu4DF32(a.to_view() * b.to_view()),
            (Val::Cpu3DF32(a), Val::Cpu2DF32(b)) => Val::Cpu3DF32(a.to_view() * b.to_view()),
            (Val::Cpu4DF32(a), Val::Cpu2DF32(b)) => Val::Cpu4DF32(a.to_view() * b.to_view()),
            (Val::Cpu4DF32(a), Val::Cpu3DF32(b)) => Val::Cpu4DF32(a.to_view() * b.to_view()),

            (Val::Cpu1DF64(a), Val::Cpu1DF64(b)) => Val::Cpu1DF64(a.to_view() * b.to_view()),
            (Val::Cpu2DF64(a), Val::Cpu2DF64(b)) => Val::Cpu2DF64(a.to_view() * b.to_view()),
            (Val::Cpu3DF64(a), Val::Cpu3DF64(b)) => Val::Cpu3DF64(a.to_view() * b.to_view()),
            (Val::Cpu4DF64(a), Val::Cpu4DF64(b)) => Val::Cpu4DF64(a.to_view() * b.to_view()),

            (Val::Cpu1DF64(a), Val::F64(b)) => Val::Cpu1DF64(a.to_view() * b),
            (Val::Cpu2DF64(a), Val::F64(b)) => Val::Cpu2DF64(a.to_view() * b),
            (Val::Cpu3DF64(a), Val::F64(b)) => Val::Cpu3DF64(a.to_view() * b),
            (Val::Cpu4DF64(a), Val::F64(b)) => Val::Cpu4DF64(a.to_view() * b),

            (Val::Cpu2DF64(a), Val::Cpu1DF64(b)) => Val::Cpu2DF64(a.to_view() * b.to_view()),
            (Val::Cpu3DF64(a), Val::Cpu1DF64(b)) => Val::Cpu3DF64(a.to_view() * b.to_view()),
            (Val::Cpu4DF64(a), Val::Cpu1DF64(b)) => Val::Cpu4DF64(a.to_view() * b.to_view()),
            (Val::Cpu3DF64(a), Val::Cpu2DF64(b)) => Val::Cpu3DF64(a.to_view() * b.to_view()),
            (Val::Cpu4DF64(a), Val::Cpu2DF64(b)) => Val::Cpu4DF64(a.to_view() * b.to_view()),
            (Val::Cpu4DF64(a), Val::Cpu3DF64(b)) => Val::Cpu4DF64(a.to_view() * b.to_view()),
            _ => panic!("type mismatch"),
        }
    }
}

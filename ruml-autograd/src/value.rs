use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
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
    F64(f64),
    Cpu1DF32(CpuOwnedMatrix1D<f32>),
    Cpu2DF32(CpuOwnedMatrix2D<f32>),
    Cpu3DF32(CpuOwnedMatrix3D<f32>),
    Cpu4DF32(CpuOwnedMatrix4D<f32>),

    F32(f32),
    Cpu1DF64(CpuOwnedMatrix1D<f64>),
    Cpu2DF64(CpuOwnedMatrix2D<f64>),
    Cpu3DF64(CpuOwnedMatrix3D<f64>),
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
            Val::Cpu1DF32(v) => v.shape().into_iter().collect(),
            Val::Cpu2DF32(v) => v.shape().into_iter().collect(),
            Val::Cpu3DF32(v) => v.shape().into_iter().collect(),
            Val::Cpu4DF32(v) => v.shape().into_iter().collect(),

            Val::F64(_) => vec![],
            Val::Cpu1DF64(v) => v.shape().into_iter().collect(),
            Val::Cpu2DF64(v) => v.shape().into_iter().collect(),
            Val::Cpu3DF64(v) => v.shape().into_iter().collect(),
            Val::Cpu4DF64(v) => v.shape().into_iter().collect(),
        }
    }

    pub fn num_dim(&self) -> usize {
        self.dim().len()
    }

    pub fn is_f32_matrix(&self) -> bool {
        matches!(
            self,
            Val::Cpu1DF32(_) | Val::Cpu2DF32(_) | Val::Cpu3DF32(_) | Val::Cpu4DF32(_)
        )
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

macro_rules! impl_from_trait {
    ($ty:ty, $arm:ident) => {
        impl From<$ty> for Val {
            fn from(v: $ty) -> Self {
                Val::$arm(v)
            }
        }
    };
}
impl_from_trait!(f32, F32);
impl_from_trait!(f64, F64);
impl_from_trait!(CpuOwnedMatrix1D<f32>, Cpu1DF32);
impl_from_trait!(CpuOwnedMatrix1D<f64>, Cpu1DF64);
impl_from_trait!(CpuOwnedMatrix2D<f32>, Cpu2DF32);
impl_from_trait!(CpuOwnedMatrix2D<f64>, Cpu2DF64);
impl_from_trait!(CpuOwnedMatrix3D<f32>, Cpu3DF32);
impl_from_trait!(CpuOwnedMatrix3D<f64>, Cpu3DF64);
impl_from_trait!(CpuOwnedMatrix4D<f32>, Cpu4DF32);
impl_from_trait!(CpuOwnedMatrix4D<f64>, Cpu4DF64);

macro_rules! impl_try_from_trait {
    ($ty:ty, $arm:ident) => {
        impl TryFrom<Val> for $ty {
            type Error = &'static str;
            fn try_from(v: Val) -> Result<Self, Self::Error> {
                match v {
                    Val::$arm(v) => Ok(v),
                    _ => Err("type mismatch"),
                }
            }
        }
    };
}
impl_try_from_trait!(f32, F32);
impl_try_from_trait!(f64, F64);
impl_try_from_trait!(CpuOwnedMatrix1D<f32>, Cpu1DF32);
impl_try_from_trait!(CpuOwnedMatrix1D<f64>, Cpu1DF64);
impl_try_from_trait!(CpuOwnedMatrix2D<f32>, Cpu2DF32);
impl_try_from_trait!(CpuOwnedMatrix2D<f64>, Cpu2DF64);
impl_try_from_trait!(CpuOwnedMatrix3D<f32>, Cpu3DF32);
impl_try_from_trait!(CpuOwnedMatrix3D<f64>, Cpu3DF64);
impl_try_from_trait!(CpuOwnedMatrix4D<f32>, Cpu4DF32);
impl_try_from_trait!(CpuOwnedMatrix4D<f64>, Cpu4DF64);

macro_rules! impl_std_ops {
    ($trait:ident, $method:ident, $token:tt) => {
        impl $trait<Val> for Val {
            type Output = Val;
            fn $method(self, rhs: Val) -> Self::Output {

                match (self, rhs) {
                    (Val::F32(a), Val::F32(b)) => Val::F32(a $token b),
                    (Val::F64(a), Val::F64(b)) => Val::F64(a $token b),

                    (Val::Cpu1DF32(a), Val::Cpu1DF32(b)) => Val::Cpu1DF32(a.to_view() $token b.to_view()),
                    (Val::Cpu2DF32(a), Val::Cpu2DF32(b)) => Val::Cpu2DF32(a.to_view() $token b.to_view()),
                    (Val::Cpu3DF32(a), Val::Cpu3DF32(b)) => Val::Cpu3DF32(a.to_view() $token b.to_view()),
                    (Val::Cpu4DF32(a), Val::Cpu4DF32(b)) => Val::Cpu4DF32(a.to_view() $token b.to_view()),

                    (Val::Cpu1DF32(a), Val::F32(b)) => Val::Cpu1DF32(a.to_view() $token b),
                    (Val::Cpu2DF32(a), Val::F32(b)) => Val::Cpu2DF32(a.to_view() $token b),
                    (Val::Cpu3DF32(a), Val::F32(b)) => Val::Cpu3DF32(a.to_view() $token b),
                    (Val::Cpu4DF32(a), Val::F32(b)) => Val::Cpu4DF32(a.to_view() $token b),

                    (Val::Cpu2DF32(a), Val::Cpu1DF32(b)) | (Val::Cpu1DF32(b), Val::Cpu2DF32(a)) => {
                        Val::Cpu2DF32(a.to_view() $token b.to_view())
                    }
                    (Val::Cpu3DF32(a), Val::Cpu1DF32(b)) | (Val::Cpu1DF32(b), Val::Cpu3DF32(a)) => {
                        Val::Cpu3DF32(a.to_view() $token b.to_view())
                    }
                    (Val::Cpu4DF32(a), Val::Cpu1DF32(b)) | (Val::Cpu1DF32(b), Val::Cpu4DF32(a)) => {
                        Val::Cpu4DF32(a.to_view() $token b.to_view())
                    }
                    (Val::Cpu3DF32(a), Val::Cpu2DF32(b)) | (Val::Cpu2DF32(b), Val::Cpu3DF32(a)) => {
                        Val::Cpu3DF32(a.to_view() $token b.to_view())
                    }
                    (Val::Cpu4DF32(a), Val::Cpu2DF32(b)) | (Val::Cpu2DF32(b), Val::Cpu4DF32(a)) => {
                        Val::Cpu4DF32(a.to_view() $token b.to_view())
                    }
                    (Val::Cpu4DF32(a), Val::Cpu3DF32(b)) | (Val::Cpu3DF32(b), Val::Cpu4DF32(a)) => {
                        Val::Cpu4DF32(a.to_view() $token b.to_view())
                    }

                    (Val::Cpu1DF64(a), Val::Cpu1DF64(b)) => Val::Cpu1DF64(a.to_view() $token b.to_view()),
                    (Val::Cpu2DF64(a), Val::Cpu2DF64(b)) => Val::Cpu2DF64(a.to_view() $token b.to_view()),
                    (Val::Cpu3DF64(a), Val::Cpu3DF64(b)) => Val::Cpu3DF64(a.to_view() $token b.to_view()),
                    (Val::Cpu4DF64(a), Val::Cpu4DF64(b)) => Val::Cpu4DF64(a.to_view() $token b.to_view()),

                    (Val::Cpu1DF64(a), Val::F64(b)) => Val::Cpu1DF64(a.to_view() $token b),
                    (Val::Cpu2DF64(a), Val::F64(b)) => Val::Cpu2DF64(a.to_view() $token b),
                    (Val::Cpu3DF64(a), Val::F64(b)) => Val::Cpu3DF64(a.to_view() $token b),
                    (Val::Cpu4DF64(a), Val::F64(b)) => Val::Cpu4DF64(a.to_view() $token b),

                    (Val::Cpu2DF64(a), Val::Cpu1DF64(b)) | (Val::Cpu1DF64(b), Val::Cpu2DF64(a)) => {
                        Val::Cpu2DF64(a.to_view() $token b.to_view())
                    }
                    (Val::Cpu3DF64(a), Val::Cpu1DF64(b)) | (Val::Cpu1DF64(b), Val::Cpu3DF64(a)) => {
                        Val::Cpu3DF64(a.to_view() $token b.to_view())
                    }
                    (Val::Cpu4DF64(a), Val::Cpu1DF64(b)) | (Val::Cpu1DF64(b), Val::Cpu4DF64(a)) => {
                        Val::Cpu4DF64(a.to_view() $token b.to_view())
                    }
                    (Val::Cpu3DF64(a), Val::Cpu2DF64(b)) | (Val::Cpu2DF64(b), Val::Cpu3DF64(a)) => {
                        Val::Cpu3DF64(a.to_view() $token b.to_view())
                    }
                    (Val::Cpu4DF64(a), Val::Cpu2DF64(b)) | (Val::Cpu2DF64(b), Val::Cpu4DF64(a)) => {
                        Val::Cpu4DF64(a.to_view() $token b.to_view())
                    }
                    (Val::Cpu4DF64(a), Val::Cpu3DF64(b)) | (Val::Cpu3DF64(b), Val::Cpu4DF64(a)) => {
                        Val::Cpu4DF64(a.to_view() $token b.to_view())
                    }
                    _ => panic!("type mismatch"),
                }
            }
        }
    };
}
impl_std_ops!(Add, add, +);
impl_std_ops!(Mul, mul, *);

impl Sub<Val> for Val {
    type Output = Val;

    fn sub(self, rhs: Val) -> Self::Output {
        if rhs.is_f32() || rhs.is_f32_matrix() {
            let rhs = rhs * Val::F32(-1.0);
            self + rhs
        } else {
            let rhs = rhs * Val::F64(-1.0);
            self + rhs
        }
    }
}

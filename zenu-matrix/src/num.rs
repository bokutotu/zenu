use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use rand_distr::{num_traits::Float, uniform::SampleUniform};

pub trait Num:
    Default
    + Clone
    + Copy
    + Debug
    + Display
    + Add<Self, Output = Self>
    + PartialOrd
    + Mul<Output = Self>
    + Div<Output = Self>
    + Sub<Output = Self>
    + SubAssign
    + DivAssign
    + AddAssign
    + MulAssign
    + Float
    + SampleUniform
    + 'static
{
    fn is_f32() -> bool;
    fn minus_one() -> Self;
    fn from_usize(n: usize) -> Self;
}

impl Num for f32 {
    fn is_f32() -> bool {
        true
    }

    fn minus_one() -> f32 {
        -1.0
    }

    fn from_usize(n: usize) -> f32 {
        n as f32
    }
}
impl Num for f64 {
    fn is_f32() -> bool {
        false
    }

    fn minus_one() -> f64 {
        -1.0
    }

    fn from_usize(n: usize) -> Self {
        n as f64
    }
}

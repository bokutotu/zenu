use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign},
};

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
    + DivAssign
    + AddAssign
    + MulAssign
    + 'static
{
    fn is_f32() -> bool;
    fn zero() -> Self;
    fn one() -> Self;
    fn minus_one() -> Self;
    fn exp(self) -> Self;
}

impl Num for f32 {
    fn is_f32() -> bool {
        true
    }

    fn zero() -> f32 {
        0.0
    }

    fn one() -> f32 {
        1.0
    }

    fn minus_one() -> f32 {
        -1.0
    }

    fn exp(self) -> f32 {
        self.exp()
    }
}
impl Num for f64 {
    fn is_f32() -> bool {
        false
    }

    fn zero() -> f64 {
        0.0
    }

    fn one() -> f64 {
        1.0
    }

    fn minus_one() -> f64 {
        -1.0
    }

    fn exp(self) -> f64 {
        self.exp()
    }
}

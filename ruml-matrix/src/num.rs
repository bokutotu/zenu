use std::{
    fmt::{Debug, Display},
    ops::Add,
};

pub trait Num:
    Default + Clone + Copy + Debug + Display + Add<Self, Output = Self> + PartialOrd
{
    fn is_f32() -> bool;
    fn zero() -> Self;
    fn one() -> Self;
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
}

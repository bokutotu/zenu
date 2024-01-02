use std::fmt::{Debug, Display};

pub trait Num: Default + Clone + Copy + Debug + Display {
    fn is_f32() -> bool;
    fn zero() -> Self;
}

impl Num for f32 {
    fn is_f32() -> bool {
        true
    }

    fn zero() -> f32 {
        0.0
    }
}
impl Num for f64 {
    fn is_f32() -> bool {
        false
    }

    fn zero() -> f64 {
        0.0
    }
}

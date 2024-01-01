use std::fmt::{Debug, Display};

pub trait Num: Default + Clone + Copy + Debug + Display {
    fn is_f32() -> bool;
}

impl Num for f32 {
    fn is_f32() -> bool {
        true
    }
}
impl Num for f64 {
    fn is_f32() -> bool {
        false
    }
}

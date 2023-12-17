use std::fmt::Debug;

pub trait Num: Default + Clone + Copy + Debug {
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

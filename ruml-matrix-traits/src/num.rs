use std::fmt::Debug;

pub trait Num: Default + Clone + Copy + Debug {}

impl Num for f32 {}
impl Num for f64 {}

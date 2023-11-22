pub trait Num: Default + Clone + Copy {}

impl Num for f32 {}
impl Num for f64 {}

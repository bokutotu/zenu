use std::fmt::Debug;

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
    fn one() -> Self;
}

pub trait Value:
    Zero + One + Clone + Debug + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> + 'static
{
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

impl One for f64 {
    fn one() -> Self {
        1.0
    }
}

impl Value for f32 {}

impl Value for f64 {}

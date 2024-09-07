//! Constructors for creating random matrices.
//!
//! This module provides functions and builders for creating matrices filled with random values
//! from various distributions such as normal distribution and uniform distribution.

use std::marker::PhantomData;

use crate::{
    device::DeviceBase,
    dim::DimTrait,
    matrix::{Matrix, Owned, Repr},
    num::Num,
};
use rand::prelude::*;
use rand_distr::{num_traits::Float, uniform::SampleUniform, Normal, StandardNormal, Uniform};

/// Creates a matrix filled with random values from a normal distribution.
///
/// # Arguments
///
/// * `mean` - The mean of the normal distribution.
/// * `std_dev` - The standard deviation of the normal distribution.
/// * `shape` - The shape of the matrix.
/// * `seed` - An optional seed for the random number generator.
/// # Panics
/// Normal distribution may fail to create if the standard deviation is negative.
pub fn normal<T: Num, S: DimTrait, D: DeviceBase>(
    mean: T,
    std_dev: T,
    shape: S,
    seed: Option<u64>,
) -> Matrix<Owned<T>, S, D>
where
    StandardNormal: Distribution<T>,
{
    let mut rng: Box<dyn RngCore> = if let Some(seed) = seed {
        Box::new(StdRng::seed_from_u64(seed))
    } else {
        Box::new(thread_rng())
    };
    let normal = Normal::new(mean, std_dev).unwrap();
    let mut data = Vec::with_capacity(shape.num_elm());
    for _ in 0..shape.num_elm() {
        data.push(normal.sample(&mut *rng));
    }
    Matrix::from_vec(data, shape)
}

/// Creates a matrix filled with random values from a normal distribution with the same shape as another matrix.
///
/// # Arguments
///
/// * `mean` - The mean of the normal distribution.
/// * `std_dev` - The standard deviation of the normal distribution.
/// * `a` - The matrix whose shape is used.
/// * `seed` - An optional seed for the random number generator.
pub fn normal_like<T: Num, S: DimTrait, D: DeviceBase>(
    mean: T,
    std_dev: T,
    a: &Matrix<Owned<T>, S, D>,
    seed: Option<u64>,
) -> Matrix<Owned<T>, S, D>
where
    StandardNormal: Distribution<T>,
{
    normal(mean, std_dev, a.shape(), seed)
}

/// Creates a matrix filled with random values from a uniform distribution.
///
/// # Arguments
///
/// * `low` - The lower bound of the uniform distribution.
/// * `high` - The upper bound of the uniform distribution.
/// * `shape` - The shape of the matrix.
/// * `seed` - An optional seed for the random number generator.
pub fn uniform<T, S: DimTrait, D: DeviceBase>(
    low: T,
    high: T,
    shape: S,
    seed: Option<u64>,
) -> Matrix<Owned<T>, S, D>
where
    T: Num,
    Uniform<T>: Distribution<T>,
{
    let mut rng: Box<dyn RngCore> = if let Some(seed) = seed {
        Box::new(StdRng::seed_from_u64(seed))
    } else {
        Box::new(thread_rng())
    };
    let uniform = Uniform::new(low, high);
    let mut data = Vec::with_capacity(shape.num_elm());
    for _ in 0..shape.num_elm() {
        data.push(uniform.sample(&mut *rng));
    }
    Matrix::from_vec(data, shape)
}

/// Creates a matrix filled with random values from a uniform distribution with the same shape as another matrix.
///
/// # Arguments
///
/// * `low` - The lower bound of the uniform distribution.
/// * `high` - The upper bound of the uniform distribution.
/// * `a` - The matrix whose shape is used.
/// * `seed` - An optional seed for the random number generator.
pub fn uniform_like<T, S: DimTrait, D: DeviceBase>(
    low: T,
    high: T,
    a: &Matrix<Owned<T>, S, D>,
    seed: Option<u64>,
) -> Matrix<Owned<T>, S, D>
where
    T: Num,
    Uniform<T>: Distribution<T>,
{
    uniform(low, high, a.shape(), seed)
}

/// A builder for creating matrices filled with random values from a normal distribution.
#[derive(Debug, Clone, Default)]
pub struct NormalBuilder<T: Num + Float, S: DimTrait, D: DeviceBase> {
    mean: Option<T>,
    std_dev: Option<T>,
    shape: Option<S>,
    seed: Option<u64>,
    _marker: PhantomData<D>,
}

impl<T, S, D> NormalBuilder<T, S, D>
where
    T: Num,
    S: DimTrait,
    D: DeviceBase,
{
    /// Creates a new `NormalBuilder`.
    #[must_use]
    pub fn new() -> Self {
        NormalBuilder {
            mean: None,
            std_dev: None,
            shape: None,
            seed: None,
            _marker: PhantomData,
        }
    }

    /// Sets the mean of the normal distribution.
    #[must_use]
    pub fn mean(mut self, mean: T) -> Self {
        self.mean = Some(mean);
        self
    }

    /// Sets the standard deviation of the normal distribution.
    #[must_use]
    pub fn std_dev(mut self, std_dev: T) -> Self {
        self.std_dev = Some(std_dev);
        self
    }

    /// Sets the shape of the matrix.
    #[must_use]
    pub fn shape(mut self, shape: S) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Sets the seed for the random number generator.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the shape of the matrix to be the same as another matrix.
    #[must_use]
    pub fn from_matrx<R2: Repr<Item = T>>(mut self, a: &Matrix<R2, S, D>) -> Self {
        self.shape = Some(a.shape());
        self
    }

    /// Builds the matrix.
    /// # Panics
    /// `mean` and `std_dev` and `shape` is not set.
    #[must_use]
    pub fn build(self) -> Matrix<Owned<T>, S, D>
    where
        StandardNormal: Distribution<T>,
    {
        assert!(self.mean.is_some(), "mean must be set");
        assert!(self.std_dev.is_some(), "std_dev must be set");
        assert!(self.shape.is_some(), "shape must be set");

        normal(
            self.mean.unwrap(),
            self.std_dev.unwrap(),
            self.shape.unwrap(),
            self.seed,
        )
    }
}

/// A builder for creating matrices filled with random values from a uniform distribution.
#[derive(Debug, Clone, Default)]
pub struct UniformBuilder<T, S, D> {
    low: Option<T>,
    high: Option<T>,
    shape: Option<S>,
    seed: Option<u64>,
    _marker: PhantomData<D>,
}

impl<T, S, D> UniformBuilder<T, S, D>
where
    T: Num + Float + SampleUniform,
    Uniform<T>: Distribution<T>,
    S: DimTrait,
    D: DeviceBase,
{
    /// Creates a new `UniformBuilder`.
    #[must_use]
    pub fn new() -> Self {
        UniformBuilder {
            low: None,
            high: None,
            shape: None,
            seed: None,
            _marker: PhantomData,
        }
    }

    /// Sets the lower bound of the uniform distribution.
    #[must_use]
    pub fn low(mut self, low: T) -> Self {
        self.low = Some(low);
        self
    }

    /// Sets the upper bound of the uniform distribution.
    #[must_use]
    pub fn high(mut self, high: T) -> Self {
        self.high = Some(high);
        self
    }

    /// Sets the shape of the matrix.
    #[must_use]
    pub fn shape(mut self, shape: S) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Sets the seed for the random number generator.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the shape of the matrix to be the same as another matrix.
    #[must_use]
    pub fn from_matrx<R2: Repr<Item = T>>(mut self, a: &Matrix<R2, S, D>) -> Self {
        self.shape = Some(a.shape());
        self
    }

    /// Builds the matrix.
    /// # Panics
    /// `low`, `high`, and `shape` is not set.
    pub fn build(self) -> Matrix<Owned<T>, S, D> {
        // if self.low.is_none() || self.high.is_none() || self.shape.is_none() {
        //     panic!("low, high, and shape must be set");
        // }
        assert!(self.low.is_some(), "low must be set");
        assert!(self.high.is_some(), "high must be set");
        assert!(self.shape.is_some(), "shape must be set");

        uniform(
            self.low.unwrap(),
            self.high.unwrap(),
            self.shape.unwrap(),
            self.seed,
        )
    }
}

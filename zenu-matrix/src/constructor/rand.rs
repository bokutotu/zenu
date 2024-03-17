//! Constructors for creating random matrices.
//!
//! This module provides functions and builders for creating matrices filled with random values
//! from various distributions such as normal distribution and uniform distribution.

use crate::{
    dim::DimTrait,
    matrix::{MatrixBase, OwnedMatrix},
    matrix_impl::Matrix,
    memory_impl::OwnedMem,
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
pub fn normal<T: Num, D: DimTrait>(
    mean: T,
    std_dev: T,
    shape: D,
    seed: Option<u64>,
) -> Matrix<OwnedMem<T>, D>
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
pub fn normal_like<T: Num, D: DimTrait>(
    mean: T,
    std_dev: T,
    a: &Matrix<OwnedMem<T>, D>,
    seed: Option<u64>,
) -> Matrix<OwnedMem<T>, D>
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
pub fn uniform<T, D: DimTrait>(
    low: T,
    high: T,
    shape: D,
    seed: Option<u64>,
) -> Matrix<OwnedMem<T>, D>
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
pub fn uniform_like<T, D: DimTrait>(
    low: T,
    high: T,
    a: &Matrix<OwnedMem<T>, D>,
    seed: Option<u64>,
) -> Matrix<OwnedMem<T>, D>
where
    T: Num,
    Uniform<T>: Distribution<T>,
{
    uniform(low, high, a.shape(), seed)
}

/// A builder for creating matrices filled with random values from a normal distribution.
#[derive(Debug, Clone, Default)]
pub struct NormalBuilder<T: Num + Float, D: DimTrait> {
    mean: Option<T>,
    std_dev: Option<T>,
    shape: Option<D>,
    seed: Option<u64>,
}

impl<T, D> NormalBuilder<T, D>
where
    T: Num,
    D: DimTrait,
{
    /// Creates a new `NormalBuilder`.
    pub fn new() -> Self {
        NormalBuilder {
            mean: None,
            std_dev: None,
            shape: None,
            seed: None,
        }
    }

    /// Sets the mean of the normal distribution.
    pub fn mean(mut self, mean: T) -> Self {
        self.mean = Some(mean);
        self
    }

    /// Sets the standard deviation of the normal distribution.
    pub fn std_dev(mut self, std_dev: T) -> Self {
        self.std_dev = Some(std_dev);
        self
    }

    /// Sets the shape of the matrix.
    pub fn shape(mut self, shape: D) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Sets the seed for the random number generator.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the shape of the matrix to be the same as another matrix.
    pub fn from_matrix<M: MatrixBase<Dim = D>>(mut self, a: &M) -> Self {
        self.shape = Some(a.shape());
        self
    }

    /// Builds the matrix.
    pub fn build(self) -> Matrix<OwnedMem<T>, D>
    where
        StandardNormal: Distribution<T>,
    {
        if self.mean.is_none() || self.std_dev.is_none() || self.shape.is_none() {
            panic!("mean, std_dev, and shape must be set");
        }
        normal(
            self.mean.unwrap(),
            self.std_dev.unwrap(),
            self.shape.unwrap(),
            self.seed,
        )
    }
}

/// A builder for creating matrices filled with random values from a uniform distribution.
pub struct UniformBuilder<T, D> {
    low: Option<T>,
    high: Option<T>,
    shape: Option<D>,
    seed: Option<u64>,
}

impl<T, D> UniformBuilder<T, D>
where
    T: Num + Float + SampleUniform,
    Uniform<T>: Distribution<T>,
    D: DimTrait,
{
    /// Creates a new `UniformBuilder`.
    pub fn new() -> Self {
        UniformBuilder {
            low: None,
            high: None,
            shape: None,
            seed: None,
        }
    }

    /// Sets the lower bound of the uniform distribution.
    pub fn low(mut self, low: T) -> Self {
        self.low = Some(low);
        self
    }

    /// Sets the upper bound of the uniform distribution.
    pub fn high(mut self, high: T) -> Self {
        self.high = Some(high);
        self
    }

    /// Sets the shape of the matrix.
    pub fn shape(mut self, shape: D) -> Self {
        self.shape = Some(shape);
        self
    }

    /// Sets the seed for the random number generator.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the shape of the matrix to be the same as another matrix.
    pub fn from_matrix<M: MatrixBase<Dim = D>>(mut self, a: &M) -> Self {
        self.shape = Some(a.shape());
        self
    }

    /// Builds the matrix.
    pub fn build(self) -> Matrix<OwnedMem<T>, D> {
        if self.low.is_none() || self.high.is_none() || self.shape.is_none() {
            panic!("low, high, and shape must be set");
        }
        uniform(
            self.low.unwrap(),
            self.high.unwrap(),
            self.shape.unwrap(),
            self.seed,
        )
    }
}

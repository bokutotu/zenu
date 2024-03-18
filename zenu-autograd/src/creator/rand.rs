use rand_distr::{Distribution, StandardNormal};
use zenu_matrix::{
    constructor::rand::{NormalBuilder, UniformBuilder},
    dim::{DimDyn, DimTrait},
    matrix::MatrixBase,
    num::Num,
};

use crate::Variable;

pub fn uniform<T: Num, D: DimTrait>(low: T, high: T, seed: Option<u64>, shape: D) -> Variable<T> {
    let mut builder = UniformBuilder::new().low(low).high(high).shape(shape);
    if let Some(seed) = seed {
        builder = builder.seed(seed);
    }
    let matrix = builder.build();
    let matrix = matrix.into_dyn_dim();
    Variable::from(matrix)
}

pub fn uniform_like<T: Num, D: DimTrait>(
    a: &Variable<T>,
    low: T,
    high: T,
    seed: Option<u64>,
) -> Variable<T> {
    uniform(low, high, seed, a.get_data().shape())
}

pub fn normal<T, I>(mean: T, std_dev: T, seed: Option<u64>, shape: I) -> Variable<T>
where
    T: Num,
    I: Into<DimDyn>,
    StandardNormal: Distribution<T>,
{
    let mut builder = NormalBuilder::new()
        .std_dev(std_dev)
        .mean(mean)
        .shape(shape.into());
    if let Some(seed) = seed {
        builder = builder.seed(seed);
    }
    let matrix = builder.build();
    let matrix = matrix.into_dyn_dim();
    Variable::from(matrix)
}

pub fn normal_like<T: Num, D: DimTrait>(
    a: &Variable<T>,
    mean: T,
    std_dev: T,
    seed: Option<u64>,
) -> Variable<T>
where
    StandardNormal: Distribution<T>,
{
    normal(mean, std_dev, seed, a.get_data().shape())
}

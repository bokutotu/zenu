use rand_distr::{Distribution, StandardNormal};
use zenu_matrix::{
    constructor::rand::{NormalBuilder, UniformBuilder},
    device::Device,
    dim::{DimDyn, DimTrait},
    num::Num,
};

use crate::Variable;

pub fn uniform<T: Num, S: DimTrait, D: Device>(
    low: T,
    high: T,
    seed: Option<u64>,
    shape: S,
) -> Variable<T, D> {
    let mut builder = UniformBuilder::new().low(low).high(high).shape(shape);
    if let Some(seed) = seed {
        builder = builder.seed(seed);
    }
    let matrix = builder.build();
    let matrix = matrix.into_dyn_dim();
    Variable::from(matrix)
}

pub fn uniform_like<T: Num, S: DimTrait, D: Device>(
    a: &Variable<T, D>,
    low: T,
    high: T,
    seed: Option<u64>,
) -> Variable<T, D> {
    uniform(low, high, seed, a.get_data().shape())
}

pub fn normal<T, I, D>(mean: T, std_dev: T, seed: Option<u64>, shape: I) -> Variable<T, D>
where
    T: Num,
    I: Into<DimDyn>,
    StandardNormal: Distribution<T>,
    D: Device,
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

pub fn normal_like<T: Num, D: Device>(
    a: &Variable<T, D>,
    mean: T,
    std_dev: T,
    seed: Option<u64>,
) -> Variable<T, D>
where
    StandardNormal: Distribution<T>,
{
    normal(mean, std_dev, seed, a.get_data().shape())
}

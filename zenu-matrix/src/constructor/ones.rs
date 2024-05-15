use crate::{
    device::DeviceBase,
    dim::DimTrait,
    matrix::{Matrix, Owned, Repr},
    num::Num,
};

impl<T, S, D> Matrix<Owned<T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: DeviceBase,
{
    pub fn ones<I: Into<S>>(dim: I) -> Self {
        let dim = dim.into();
        let data = vec![T::one(); dim.num_elm()];
        let vec = data.iter().map(|_| T::from_usize(1)).collect();
        Self::from_vec(vec, dim)
    }

    pub fn ones_like<R: Repr<Item = T>>(m: &Matrix<R, S, D>) -> Self {
        Self::ones(m.shape())
    }
}

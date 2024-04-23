use crate::{
    device::Device,
    dim::DimTrait,
    matrix::{Matrix, Owned, Repr},
    num::Num,
};

impl<T, S, D> Matrix<Owned<T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: Device,
{
    pub fn zeros<I: Into<S>>(dim: I) -> Self {
        let dim = dim.into();
        let data = vec![T::zero(); dim.num_elm()];
        let vec = data.iter().map(|_| T::from_usize(0)).collect();
        Self::from_vec(vec, dim)
    }

    pub fn zeros_like<R: Repr<Item = T>>(m: &Matrix<R, S, D>) -> Self {
        Self::zeros(m.shape())
    }
}

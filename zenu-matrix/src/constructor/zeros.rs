use crate::{
    dim::DimTrait,
    matrix::{MatrixBase, OwnedMatrix},
    num::Num,
};

pub trait Zeros: MatrixBase {
    fn zeros<I: Into<Self::Dim>>(dim: I) -> Self;
    fn zeros_like<M: MatrixBase>(m: M) -> Self;
}
impl<T, D, OM> Zeros for OM
where
    T: Num,
    D: DimTrait,
    OM: OwnedMatrix + MatrixBase<Dim = D, Item = T>,
{
    fn zeros<I: Into<Self::Dim>>(dim: I) -> Self {
        let dim = dim.into();
        let num_elm = dim.num_elm();
        let data = vec![T::zero(); num_elm];
        <Self as OwnedMatrix>::from_vec(data, dim)
    }

    fn zeros_like<M: MatrixBase>(m: M) -> Self {
        Self::zeros(m.shape().slice())
    }
}

#[cfg(test)]
mod zeros {
    use crate::matrix_impl::OwnedMatrix0D;

    use super::Zeros;

    #[test]
    fn zeros_scalar() {
        let x: OwnedMatrix0D<f32> = Zeros::zeros([]);
        assert_eq!(x.get_value(), 0.0);
    }
}

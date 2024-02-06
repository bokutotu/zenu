use crate::{
    dim::DimTrait,
    matrix::{MatrixBase, OwnedMatrix},
    num::Num,
};

pub trait Zeros: MatrixBase {
    fn zeros(dim: Self::Dim) -> Self;
}
impl<T, D, OM> Zeros for OM
where
    T: Num,
    D: DimTrait,
    OM: OwnedMatrix + MatrixBase<Dim = D, Item = T>,
{
    fn zeros(dim: D) -> Self {
        let num_elm = dim.num_elm();
        let mut data = Vec::with_capacity(num_elm);
        for _ in 0..num_elm {
            data.push(T::zero());
        }
        <Self as OwnedMatrix>::from_vec(data, dim)
    }
}

#[cfg(test)]
mod zeros {
    use crate::{dim, matrix_impl::CpuOwnedMatrix0D};

    use super::Zeros;

    #[test]
    fn zeros_scalar() {
        let x: CpuOwnedMatrix0D<f32> = Zeros::zeros(dim![]);
        assert_eq!(x.get_value(), 0.0);
    }
}

use crate::{
    device::DeviceBase,
    dim::{default_stride, DimTrait},
    matrix::{Matrix, Owned, Ptr, Repr},
    num::Num,
};

impl<T, S, D> Matrix<Owned<T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: DeviceBase,
{
    pub fn zeros<I: Into<S>>(dim: I) -> Self {
        let dim = dim.into();
        let num_elm = dim.num_elm();
        let ptr = D::zeros(num_elm);
        let ptr = Ptr::new(ptr, num_elm, 0);
        Matrix::new(ptr, dim, default_stride(dim))
    }

    pub fn zeros_like<R: Repr<Item = T>>(m: &Matrix<R, S, D>) -> Self {
        Self::zeros(m.shape())
    }
}

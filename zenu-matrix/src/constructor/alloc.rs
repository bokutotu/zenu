use crate::{
    device::DeviceBase,
    dim::{default_stride, DimTrait},
    matrix::{Matrix, Owned, Ptr, Repr},
    num::Num,
};

impl<T, S, D> Matrix<Owned<T>, S, D>
where
    T: Num,
    D: DeviceBase,
    S: DimTrait,
{
    pub fn alloc<I: Into<S>>(shape: I) -> Self {
        let shape = shape.into();
        let num_elm = shape.num_elm();
        let bytes = num_elm * std::mem::size_of::<T>();

        let ptr = Ptr::new(D::alloc(bytes).unwrap() as *mut T, num_elm, 0);

        let stride = default_stride(shape);
        Matrix::new(ptr, shape, stride)
    }

    pub fn alloc_like<R: Repr<Item = T>>(mat: &Matrix<R, S, D>) -> Self {
        let shape = mat.shape();
        Self::alloc(shape)
    }
}

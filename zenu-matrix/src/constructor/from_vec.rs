use crate::{
    device::DeviceBase,
    dim::{default_stride, DimTrait},
    matrix::{Matrix, Owned, Ptr},
    num::Num,
};

impl<T, S, D> Matrix<Owned<T>, S, D>
where
    T: Num,
    D: DeviceBase,
    S: DimTrait,
{
    pub fn from_vec<I: Into<S>>(vec: Vec<T>, shape: I) -> Self {
        let shape = shape.into();
        if vec.len() != shape.num_elm() {
            panic!(
                "Invalid Shape, vec.len() = {}, shape.num_elm() = {}",
                vec.len(),
                shape.num_elm()
            );
        }

        let len = vec.len();

        let ptr = Ptr::new(D::from_vec(vec), len, 0);

        let stride = default_stride(shape);
        Matrix::new(ptr, shape, stride)
    }
}

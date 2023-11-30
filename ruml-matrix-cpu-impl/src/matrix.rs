use ruml_matrix_traits::{dim::DimTrait, memory::Memory};

#[derive(Clone)]
pub struct CpuMatrix<M: Memory, D: DimTrait> {
    data: M,
    shape: D,
    stride: D,
}

impl<M: Memory + Clone, D: DimTrait + Clone> CpuMatrix<M, D> {
    pub fn new(data: M, shape: D, stride: D) -> Self {
        Self {
            data,
            shape,
            stride,
        }
    }

    pub fn shape(&self) -> D {
        self.shape.clone()
    }

    pub fn stride(&self) -> D {
        self.stride.clone()
    }

    pub fn data(&self) -> &M {
        &self.data
    }
}

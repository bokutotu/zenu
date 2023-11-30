use ruml_matrix_traits::{
    dim::{default_stride, DimTrait},
    index::ShapeStride,
    matrix::{Matrix, OwnedMatrix, ViewMatrix},
    memory::{Memory, OwnedMemory, ViewMemory},
};

use crate::matrix::CpuMatrix;

impl<M: Memory + Clone, D: DimTrait + Clone> Matrix for CpuMatrix<M, D> {
    type Dim = D;
    type Memory = M;

    fn shape_stride(&self) -> ShapeStride<Self::Dim> {
        ShapeStride::new(self.shape(), self.stride())
    }

    fn construct(data: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self {
        Self::new(data, shape, stride)
    }

    fn memory(&self) -> &Self::Memory {
        &self.data()
    }
}

impl<'a, M: OwnedMemory<'a>, D: DimTrait> OwnedMatrix<'a> for CpuMatrix<M, D> {
    type View = CpuMatrix<<M as OwnedMemory<'a>>::View, D>;

    fn to_view(&'a self) -> Self::View {
        let data = self.data().to_view(0);
        let shape_stride = self.shape_stride();
        Self::View::new(data, shape_stride.shape(), shape_stride.stride())
    }

    fn from_vec(vec: Vec<<<Self as Matrix>::Memory as Memory>::Item>, dim: Self::Dim) -> Self {
        let stride = default_stride(dim);
        let data = M::from_vec(vec);
        Self::construct(data, dim, stride)
    }
}

impl<'a, M: ViewMemory<'a>, D: DimTrait> ViewMatrix<'a> for CpuMatrix<M, D> {
    type Owned = CpuMatrix<<M as ViewMemory<'a>>::Owned, D>;

    fn to_owned(&self) -> Self::Owned {
        let owned = self.data().to_owned();
        let shape_stride = self.shape_stride();
        Self::Owned::construct(owned, shape_stride.shape(), shape_stride.stride())
    }
}

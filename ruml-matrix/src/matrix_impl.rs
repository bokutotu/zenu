use crate::{
    dim::{DimTrait, LessDimTrait},
    index::{IndexAxisTrait, ShapeStride, SliceTrait},
    matrix::{
        IndexAxis, IndexAxisMut, MatrixBase, MatrixSlice, MatrixSliceMut, OwnedMatrix, ViewMatrix,
    },
    memory::{Memory, OwnedMemory, ToViewMemory, ToViewMutMemory, ViewMemory},
};

pub struct Matrix<M, S> {
    memory: M,
    shape: S,
    stride: S,
}

impl<M, S> MatrixBase for Matrix<M, S>
where
    M: Memory,
    S: DimTrait,
{
    type Memory = M;
    type Dim = S;
    fn memory(&self) -> &Self::Memory {
        &self.memory
    }

    fn shape_stride(&self) -> ShapeStride<Self::Dim> {
        ShapeStride::new(self.shape, self.stride)
    }

    fn construct(memory: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self {
        Self {
            memory,
            shape,
            stride,
        }
    }

    fn memory_mut(&mut self) -> &mut Self::Memory {
        &mut self.memory
    }
}

impl<M, S> ViewMatrix for Matrix<M, S>
where
    M: ViewMemory,
    S: DimTrait,
{
}

impl<M, S> OwnedMatrix for Matrix<M, S>
where
    M: OwnedMemory,
    S: DimTrait,
{
}

impl<M, S, SS> MatrixSlice<S, SS> for Matrix<M, S>
where
    M: ToViewMemory,
    S: DimTrait,
    SS: SliceTrait<Dim = S>,
{
}

impl<M, S, SS> MatrixSliceMut<S, SS> for Matrix<M, S>
where
    M: ToViewMutMemory,
    S: DimTrait,
    SS: SliceTrait<Dim = S>,
{
}

impl<M, S, I> IndexAxis<I> for Matrix<M, S>
where
    M: ToViewMemory,
    S: DimTrait + LessDimTrait,
    I: IndexAxisTrait,
{
    type Output<'a> = Matrix<
        <<Matrix<M, S> as MatrixBase>::Memory as ToViewMemory>::View<'a>,
        <<Matrix<M, S> as MatrixBase>::Dim as LessDimTrait>::LessDim,
    >
    where
        Self: 'a;
}

impl<M, S, I> IndexAxisMut<I> for Matrix<M, S>
where
    M: ToViewMutMemory,
    S: DimTrait + LessDimTrait,
    I: IndexAxisTrait,
{
    type Output<'a> = Matrix<
        <<Matrix<M, S> as MatrixBase>::Memory as ToViewMutMemory>::ViewMut<'a>,
        <<Matrix<M, S> as MatrixBase>::Dim as LessDimTrait>::LessDim,
    >
    where
        Self: 'a;
}

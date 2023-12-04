use crate::{
    dim::{default_stride, DimTrait, LessDimTrait},
    index::{IndexAxisTrait, ShapeStride, SliceTrait},
    memory::{Memory, OwnedMemory, ToViewMemory, ToViewMutMemory, ViewMemory},
};

pub trait MatrixBase: Sized {
    type Dim: DimTrait;
    type Memory: Memory;

    fn shape_stride(&self) -> ShapeStride<Self::Dim>;
    fn memory(&self) -> &Self::Memory;
    fn memory_mut(&mut self) -> &mut Self::Memory;
    fn construct(data: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self;
}

pub trait OwnedMatrix: MatrixBase
where
    Self::Memory: OwnedMemory,
{
    fn to_view<'a, D>(&'a self) -> D
    where
        D: MatrixBase<
            Memory = <<Self as MatrixBase>::Memory as ToViewMemory>::View<'a>,
            Dim = Self::Dim,
        >,
    {
        D::construct(
            self.memory().to_view(self.memory().get_offset()),
            self.shape_stride().shape(),
            self.shape_stride().stride(),
        )
    }

    fn to_view_mut<'a, DM>(&'a mut self) -> DM
    where
        DM: MatrixBase<
            Memory = <<Self as MatrixBase>::Memory as ToViewMutMemory>::ViewMut<'a>,
            Dim = Self::Dim,
        >,
    {
        let offset = self.memory().get_offset();
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        DM::construct(self.memory_mut().to_view_mut(offset), shape, stride)
    }

    fn from_vec(vec: Vec<<<Self as MatrixBase>::Memory as Memory>::Item>, dim: Self::Dim) -> Self {
        let stride = default_stride(dim);
        let memory = Self::Memory::from_vec(vec);
        Self::construct(memory, dim, stride)
    }
}

pub trait ViewMatrix: MatrixBase
where
    Self::Memory: ViewMemory,
{
    fn to_owned<O>(&self) -> O
    where
        O: MatrixBase<Memory = <<Self as MatrixBase>::Memory as ToOwned>::Owned, Dim = Self::Dim>,
    {
        O::construct(
            self.memory().to_owned(),
            self.shape_stride().shape(),
            self.shape_stride().stride(),
        )
    }
}

pub trait MatrixSlice<D, S>: MatrixBase<Dim = D>
where
    S: SliceTrait<Dim = D>,
    D: DimTrait,
    Self::Memory: ToViewMemory,
{
    type Output<'a>: MatrixBase<Memory = <Self::Memory as ToViewMemory>::View<'a>, Dim = D>
    where
        Self: 'a;

    fn slice(&self, index: S) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.sliced_shape_stride(shape, stride);
        let offset = index.sliced_offset(stride, self.memory().get_offset());
        Self::Output::construct(
            self.memory().to_view(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

pub trait MatrixSliceMut<D, S>: MatrixBase<Dim = D>
where
    S: SliceTrait<Dim = D>,
    D: DimTrait,
    Self::Memory: ToViewMutMemory,
{
    type Output<'a>: MatrixBase<Memory = <Self::Memory as ToViewMutMemory>::ViewMut<'a>, Dim = D>
    where
        Self: 'a;

    fn slice_mut(&mut self, index: S) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.sliced_shape_stride(shape, stride);
        let offset = index.sliced_offset(stride, self.memory().get_offset());
        Self::Output::construct(
            self.memory_mut().to_view_mut(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

pub trait IndexAxis<I: IndexAxisTrait>: MatrixBase
where
    Self::Memory: ToViewMemory,
    Self::Dim: LessDimTrait,
{
    type Output<'a>: MatrixBase<
        Memory = <Self::Memory as ToViewMemory>::View<'a>,
        Dim = <Self::Dim as LessDimTrait>::LessDim,
    >
    where
        Self: 'a;

    fn index_axis(&self, index: I) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.get_shape_stride(shape, stride);
        let offset = index.offset(stride);
        Self::Output::construct(
            self.memory().to_view(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

pub trait IndexAxisMut<I: IndexAxisTrait>: MatrixBase
where
    Self::Memory: ToViewMutMemory,
    Self::Dim: LessDimTrait,
{
    type Output<'a>: MatrixBase<
        Memory = <Self::Memory as ToViewMutMemory>::ViewMut<'a>,
        Dim = <Self::Dim as LessDimTrait>::LessDim,
    >
    where
        Self: 'a;

    fn index_axis_mut(&mut self, index: I) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.get_shape_stride(shape, stride);
        let offset = index.offset(stride);
        Self::Output::construct(
            self.memory_mut().to_view_mut(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

use crate::{
    dim::{cal_offset, default_stride, DimTrait, LessDimTrait},
    index::{IndexAxisTrait, ShapeStride, SliceTrait},
    memory::{
        Memory, OwnedMemory, ToOwnedMemory, ToViewMemory, ToViewMutMemory, ViewMemory,
        ViewMutMemory,
    },
};

pub trait MatrixBase: Sized {
    type Dim: DimTrait;
    type Memory: Memory;

    fn shape_stride(&self) -> ShapeStride<Self::Dim>;
    fn memory(&self) -> &Self::Memory;
    fn memory_mut(&mut self) -> &mut Self::Memory;
    fn construct(data: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self;
    fn is_default_stride(&self) -> bool {
        default_stride(self.shape_stride().shape()) == self.shape_stride().stride()
    }
}

pub trait OwnedMatrix<M>: MatrixBase<Memory = M>
where
    M: OwnedMemory + ToViewMemory + ToViewMutMemory,
{
    type View<'a>: MatrixBase<
        Memory = <<Self as MatrixBase>::Memory as ToViewMemory>::View<'a>,
        Dim = Self::Dim,
    >
    where
        Self: 'a,
        M: 'a;

    type ViewMut<'a>: MatrixBase<
        Memory = <<Self as MatrixBase>::Memory as ToViewMutMemory>::ViewMut<'a>,
        Dim = Self::Dim,
    >
    where
        Self: 'a,
        M: 'a;

    fn to_view(&self) -> Self::View<'_> {
        Self::View::construct(
            self.memory().to_view(self.memory().get_offset()),
            self.shape_stride().shape(),
            self.shape_stride().stride(),
        )
    }

    fn to_view_mut(&mut self) -> Self::ViewMut<'_> {
        let offset = self.memory().get_offset();
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        Self::ViewMut::construct(self.memory_mut().to_view_mut(offset), shape, stride)
    }

    fn from_vec(vec: Vec<<<Self as MatrixBase>::Memory as Memory>::Item>, dim: Self::Dim) -> Self {
        let stride = default_stride(dim);
        let memory = Self::Memory::from_vec(vec);
        Self::construct(memory, dim, stride)
    }
}

pub trait ViewMatrix<M>: MatrixBase<Memory = M>
where
    M: ViewMemory,
{
    type Owned: MatrixBase<Memory = <<Self as MatrixBase>::Memory as ToOwnedMemory>::Owned, Dim = Self::Dim>
        + OwnedMatrix<<<Self as MatrixBase>::Memory as ToOwnedMemory>::Owned>;
    fn to_owned(&self) -> Self::Owned {
        Self::Owned::construct(
            self.memory().to_owned_memory(),
            self.shape_stride().shape(),
            self.shape_stride().stride(),
        )
    }
}

pub trait ViewMutMatix<M>: MatrixBase<Memory = M>
where
    M: ViewMutMemory,
{
    fn view_mut_memory(&self) -> &Self::Memory {
        self.memory()
    }
}

pub trait MatrixSlice<M, D, S>: MatrixBase<Memory = M, Dim = D>
where
    S: SliceTrait<Dim = D>,
    D: DimTrait,
    M: ToViewMemory,
{
    type Output<'a>: MatrixBase<Memory = <Self::Memory as ToViewMemory>::View<'a>, Dim = D>
    where
        Self: 'a,
        M: 'a;

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

pub trait IndexItem<D>: MatrixBase<Dim = D>
where
    D: DimTrait,
{
    fn index_item(&self, index: D) -> <Self::Memory as Memory>::Item {
        if self.shape_stride().shape().is_overflow(index) {
            panic!("index out of bounds");
        }

        let offset = cal_offset(index, self.shape_stride().stride());
        self.memory().ptr_offset(offset)
    }
}

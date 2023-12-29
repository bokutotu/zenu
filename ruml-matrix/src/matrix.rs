use crate::{
    dim::{cal_offset, default_stride, DimTrait},
    index::{IndexAxisTrait, ShapeStride},
    memory::Memory,
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

pub trait OwnedMatrix: MatrixBase {
    type View<'a>: ViewMatrix
    where
        Self: 'a;
    type ViewMut<'a>: ViewMutMatix
    where
        Self: 'a;

    fn to_view(&self) -> Self::View<'_>;
    fn to_view_mut(&mut self) -> Self::ViewMut<'_>;
    fn from_vec(vec: Vec<<<Self as MatrixBase>::Memory as Memory>::Item>, dim: Self::Dim) -> Self;
}

pub trait ViewMatrix: MatrixBase {
    type Owned: OwnedMatrix;

    fn to_owned(&self) -> Self::Owned;
}
pub trait ViewMutMatix: MatrixBase {
    fn view_mut_memory(&self) -> &Self::Memory;
}

pub trait MatrixSlice<M, D, S>: MatrixBase {
    type Output<'a>: ViewMatrix
    where
        Self: 'a;

    fn slice(&self, index: S) -> Self::Output<'_>;
}

pub trait MatrixSliceMut<D, S>: MatrixBase<Dim = D> {
    type Output<'a>: ViewMutMatix
    where
        Self: 'a;

    fn slice_mut(&mut self, index: S) -> Self::Output<'_>;
}

pub trait IndexAxis<I: IndexAxisTrait>: MatrixBase {
    type Output<'a>: ViewMatrix
    where
        Self: 'a;

    fn index_axis(&self, index: I) -> Self::Output<'_>;
}

pub trait IndexAxisMut<I: IndexAxisTrait>: MatrixBase {
    type Output<'a>: ViewMutMatix
    where
        Self: 'a;

    fn index_axis_mut(&mut self, index: I) -> Self::Output<'_>;
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

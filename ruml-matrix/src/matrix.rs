use crate::{
    blas::Blas,
    dim::{default_stride, DimTrait, LessDimTrait},
    index::{IndexAxisTrait, ShapeStride, SliceTrait},
    num::Num,
};
pub trait MatrixBase: Sized {
    type Dim: DimTrait;
    type Item: Num;

    fn shape_stride(&self) -> ShapeStride<Self::Dim>;
    fn shape(&self) -> Self::Dim {
        self.shape_stride().shape()
    }
    fn stride(&self) -> Self::Dim {
        self.shape_stride().stride()
    }
    // fn memory(&self) -> &Self::Memory;
    // fn memory_mut(&mut self) -> &mut Self::Memory;
    // fn construct(data: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self;
    fn is_default_stride(&self) -> bool {
        default_stride(self.shape_stride().shape()) == self.shape_stride().stride()
    }
}

pub trait ToViewMatrix: MatrixBase {
    type View<'a>: ViewMatrix
    where
        Self: 'a;

    fn to_view(&self) -> Self::View<'_>;
}

pub trait ToViewMutMatrix: MatrixBase {
    type ViewMut<'a>: ViewMutMatix
    where
        Self: 'a;

    fn to_view_mut(&mut self) -> Self::ViewMut<'_>;
}

pub trait ToOwnedMatrix: MatrixBase {
    type Owned: OwnedMatrix
    where
        Self: Sized;

    fn to_owned(&self) -> Self::Owned;
}

pub trait AsMutPtr: MatrixBase {
    fn as_mut_ptr(&mut self) -> *mut Self::Item;
}

pub trait AsPtr: MatrixBase {
    fn as_ptr(&self) -> *const Self::Item;
}

pub trait MatrixSlice<S>: ToViewMatrix
where
    S: SliceTrait<Dim = Self::Dim>,
{
    type Output<'a>: MatrixBase<Dim = Self::Dim> + ViewMatrix
    where
        Self: 'a;

    fn slice(&self, index: S) -> Self::Output<'_>;
}

pub trait MatrixSliceMut<S>: ToViewMutMatrix
where
    S: SliceTrait<Dim = Self::Dim>,
{
    type Output<'a>: MatrixBase<Dim = Self::Dim> + ViewMutMatix
    where
        Self: 'a;

    fn slice_mut(&mut self, index: S) -> Self::Output<'_>;
}

pub trait IndexAxis<I: IndexAxisTrait>: ToViewMatrix
where
    Self::Dim: LessDimTrait,
{
    type Output<'a>: MatrixBase<Dim = <Self::Dim as LessDimTrait>::LessDim> + ViewMatrix
    where
        Self: 'a;

    fn index_axis(&self, index: I) -> Self::Output<'_>;
}

pub trait IndexAxisMut<I: IndexAxisTrait>: ToViewMutMatrix
where
    Self::Dim: LessDimTrait,
{
    type Output<'a>: MatrixBase<Dim = <Self::Dim as LessDimTrait>::LessDim> + ViewMutMatix
    where
        Self: 'a;

    fn index_axis_mut(&mut self, index: I) -> Self::Output<'_>;
}

pub trait IndexItem: MatrixBase {
    fn index_item(&self, index: Self::Dim) -> Self::Item;
}

pub trait IndexItemAsign: MatrixBase {
    fn index_item_asign(&mut self, index: Self::Dim, value: <Self as MatrixBase>::Item);
}

pub trait BlasMatrix: MatrixBase {
    type Blas: Blas<Self::Item>;
}

pub trait ViewMatrix:
    MatrixBase + ToViewMatrix + ToOwnedMatrix + AsPtr + BlasMatrix + IndexItem
{
}

pub trait ViewMutMatix:
    MatrixBase
    + ToViewMatrix
    + ToViewMutMatrix
    + AsMutPtr
    + BlasMatrix
    + AsPtr
    + IndexItem
    + IndexItemAsign
{
}

pub trait OwnedMatrix: MatrixBase + ToViewMatrix + ToViewMutMatrix + AsPtr + BlasMatrix {
    fn from_vec(vec: Vec<Self::Item>, dim: Self::Dim) -> Self;
}

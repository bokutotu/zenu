use crate::{
    blas::Blas,
    dim::{DimDyn, DimTrait, LessDimTrait},
    index::{IndexAxisTrait, SliceTrait},
    matrix_impl::Matrix,
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
    shape_stride::ShapeStride,
    slice::Slice,
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
        self.shape_stride().is_default_stride()
    }

    fn is_transposed_default_stride(&self) -> bool {
        self.shape_stride().is_transposed_default_stride()
    }
}

pub trait ToViewMatrix: MatrixBase {
    fn to_view(&self) -> Matrix<ViewMem<Self::Item>, Self::Dim>;
}

pub trait ToViewMutMatrix: MatrixBase {
    // type ViewMut<'a>: ViewMutMatix
    // where
    //     Self: 'a;

    fn to_view_mut(&mut self) -> Matrix<ViewMutMem<Self::Item>, Self::Dim>;
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
    type Output<'a>: MatrixBase<Dim = <Self::Dim as LessDimTrait>::LessDim, Item = Self::Item>
        + ViewMatrix
    where
        Self: 'a;

    fn index_axis(&self, index: I) -> Self::Output<'_>;
}

pub trait IndexAxisMut<I: IndexAxisTrait>: ToViewMutMatrix
where
    Self::Dim: LessDimTrait,
{
    type Output<'a>: MatrixBase<Dim = <Self::Dim as LessDimTrait>::LessDim, Item = Self::Item>
        + ViewMutMatix
    where
        Self: 'a;

    fn index_axis_mut(&mut self, index: I) -> Self::Output<'_>;
}

pub trait IndexAxisDyn<I: IndexAxisTrait>: ToViewMatrix {
    type Output<'a>: MatrixBase<Dim = DimDyn, Item = Self::Item> + ViewMatrix
    where
        Self: 'a;

    fn index_axis_dyn(&self, index: I) -> Self::Output<'_>;
}

pub trait IndexAxisMutDyn<I: IndexAxisTrait>: ToViewMutMatrix {
    type Output<'a>: MatrixBase<Dim = DimDyn, Item = Self::Item> + ViewMutMatix
    where
        Self: 'a;

    fn index_axis_mut_dyn(&mut self, index: I) -> Self::Output<'_>;
}

pub trait IndexItem: MatrixBase {
    fn index_item<I: Into<Self::Dim>>(&self, index: I) -> Self::Item;
}

pub trait IndexItemAsign: MatrixBase {
    fn index_item_asign<I: Into<Self::Dim>>(&mut self, index: I, value: <Self as MatrixBase>::Item);
}

pub trait MatrixSliceDyn: ToViewMatrix {
    type Output<'a>: MatrixBase<Dim = DimDyn> + ViewMatrix
    where
        Self: 'a;

    fn slice_dyn(&self, index: Slice) -> Self::Output<'_>;
}

pub trait MatrixSliceMutDyn: ToViewMutMatrix {
    type Output<'a>: MatrixBase<Dim = DimDyn> + ViewMutMatix
    where
        Self: 'a;

    fn slice_mut_dyn(&mut self, index: Slice) -> Self::Output<'_>;
}

pub trait BlasMatrix: MatrixBase {
    type Blas: Blas<Self::Item>;
}

pub trait ViewMatrix: MatrixBase + ToViewMatrix + ToOwnedMatrix + AsPtr + BlasMatrix {}
pub trait ViewMutMatix:
    MatrixBase + ToViewMatrix + ToViewMutMatrix + AsMutPtr + BlasMatrix + AsPtr
// + IndexItem
// + IndexItemAsign
{
}

pub trait OwnedMatrix: MatrixBase + ToViewMatrix + ToViewMutMatrix + AsPtr + BlasMatrix {
    fn from_vec<I: Into<Self::Dim>>(vec: Vec<Self::Item>, dim: I) -> Self;
}

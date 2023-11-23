use std::marker::PhantomData;

use ruml_dim_impl::{Dim0, Dim1, Dim2, Dim3, Dim4};
use ruml_matrix_traits::index::{IndexTrait, ShapeStride, SliceTrait};

pub struct DimNo1 {}
pub struct DimNo2 {}
pub struct DimNo3 {}
pub struct DimNo4 {}

pub trait DimNo {
    fn dim(&self) -> usize;
}

pub trait ForDim0: DimNo {}
pub trait ForDim1: DimNo {}
pub trait ForDim2: DimNo {}
pub trait ForDim3: DimNo {}
pub trait ForDim4: DimNo {}

impl ForDim1 for DimNo1 {}

impl ForDim2 for DimNo1 {}
impl ForDim2 for DimNo2 {}

impl ForDim3 for DimNo1 {}
impl ForDim3 for DimNo2 {}
impl ForDim3 for DimNo3 {}

impl ForDim4 for DimNo1 {}
impl ForDim4 for DimNo2 {}
impl ForDim4 for DimNo3 {}
impl ForDim4 for DimNo4 {}

impl DimNo for DimNo1 {
    fn dim(&self) -> usize {
        1
    }
}

impl DimNo for DimNo2 {
    fn dim(&self) -> usize {
        2
    }
}

impl DimNo for DimNo3 {
    fn dim(&self) -> usize {
        3
    }
}

impl DimNo for DimNo4 {
    fn dim(&self) -> usize {
        4
    }
}

pub struct Index<T: DimNo> {
    _target_dim: PhantomData<T>,
    target_index: usize,
}

impl<T: DimNo> Index<T> {
    pub fn new(target_index: usize) -> Self {
        Index {
            target_index,
            _target_dim: PhantomData,
        }
    }

    pub fn target_index(&self) -> usize {
        self.target_index
    }
}

impl<DNO: ForDim0> IndexTrait<Dim0, Dim0> for Index<DNO> {}

impl<DNO: ForDim1> IndexTrait<Dim1, Dim0> for Index<DNO> {}

impl<DNO: ForDim2> IndexTrait<Dim2, Dim1> for Index<DNO> {}

impl<DNO: ForDim3> IndexTrait<Dim3, Dim2> for Index<DNO> {}

impl<DNO: ForDim4> IndexTrait<Dim4, Dim3> for Index<DNO> {}

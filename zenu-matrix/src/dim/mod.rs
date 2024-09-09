pub mod dim_dyn;
pub mod dim_static;

pub use dim_dyn::larger_shape;
pub use dim_dyn::DimDyn;
pub(crate) use dim_dyn::{into_dyn, smaller_shape};
pub use dim_static::{Dim0, Dim1, Dim2, Dim3, Dim4};

use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

pub trait DimTrait:
    Index<usize, Output = usize>
    + IndexMut<usize>
    + IntoIterator<Item = usize>
    + Clone
    + Copy
    + Default
    + PartialEq
    + Debug
    + for<'a> From<&'a [usize]>
    + for<'a> From<&'a Self>
    + 'static
{
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn is_overflow<D: DimTrait>(&self, index: D) -> bool {
        assert!(self.len() >= index.len(), "Dimension mismatch");

        index.into_iter().zip(*self).any(|(x, y)| x >= y)
    }
    fn num_elm(&self) -> usize {
        self.into_iter().product()
    }

    fn slice(&self) -> &[usize];

    fn is_scalar(&self) -> bool {
        self.len() == 0 || self.num_elm() == 1
    }
}

pub trait LessDimTrait: DimTrait {
    type LessDim: DimTrait;

    fn remove_axis(&self, axis: usize) -> Self::LessDim {
        let mut default = DimDyn::default();
        for i in 0..self.len() {
            if i == axis {
                continue;
            }
            default.push_dim(self[i]);
        }
        Self::LessDim::from(default.slice())
    }
}

pub trait GreaterDimTrait: DimTrait {
    type GreaterDim: DimTrait;
}

#[expect(clippy::missing_panics_doc)]
pub fn cal_offset<D1: DimTrait, D2: DimTrait>(shape: D1, stride: D2) -> usize {
    assert!(shape.len() == stride.len(), "Dimension mismatch");
    shape.into_iter().zip(stride).map(|(x, y)| x * y).sum()
}

pub fn default_stride<D: DimTrait>(shape: D) -> D {
    let mut stride = shape;
    let n = shape.len();

    if n == 0 {
        return stride;
    }

    if n == 1 {
        stride[0] = 1;
        return stride;
    }

    // 最後の次元のストライドは常に1
    stride[n - 1] = 1;

    // 残りの次元に対して、後ろから前へ計算
    for i in (0..n - 1).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }

    stride
}

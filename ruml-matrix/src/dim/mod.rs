pub mod dim_dyn;
pub mod dim_static;

pub(crate) use dim_dyn::into_dyn;
pub use dim_dyn::DimDyn;
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
    + 'static
{
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn is_overflow<D: DimTrait>(&self, index: D) -> bool {
        if self.len() < index.len() {
            panic!("Dimension mismatch");
        }

        index.into_iter().zip(*self).any(|(x, y)| x >= y)
    }
    fn num_elm(&self) -> usize {
        self.into_iter().product()
    }

    fn slice(&self) -> &[usize];
}

pub trait LessDimTrait: DimTrait {
    type LessDim: DimTrait;
}

pub trait GreaterDimTrait: DimTrait {
    type GreaterDim: DimTrait;
}

pub fn cal_offset<D1: DimTrait, D2: DimTrait>(shape: D1, stride: D2) -> usize {
    if shape.len() != stride.len() {
        panic!("Dimension mismatch");
    }
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

// #[macro_export]
// macro_rules! dim {
//     () => {
//         $crate::dim::dim_static::Dim0::new()
//     };
//     ($x:expr) => {
//         $crate::dim::dim_static::Dim1::new([$x])
//     };
//     ($x:expr, $y:expr) => {
//         $crate::dim::dim_static::Dim2::new([$x, $y])
//     };
//     ($x:expr, $y:expr, $z:expr) => {
//         $crate::dim::dim_static::Dim3::new([$x, $y, $z])
//     };
//     ($x:expr, $y:expr, $z:expr, $w:expr) => {
//         $crate::dim::dim_static::Dim4::new([$x, $y, $z, $w])
//     };
// }

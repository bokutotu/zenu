use std::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use serde::{Deserialize, Serialize};

use super::{DimTrait, GreaterDimTrait, LessDimTrait};

#[derive(Clone, Debug, Default, PartialEq, Copy, Serialize, Deserialize)]
pub struct DimDyn {
    dim: [usize; 6],
    len: usize,
}
/// larger_shapeは2つのshapeのうち大きい方のshapeを返す
/// xとyを受け取り、xがlarger_shapeである場合はtrueを返す
/// xがyよりも小さい場合はfalseを返す
fn larger_shape_is_x<D1: DimTrait, D2: DimTrait>(x: D1, y: D2) -> bool {
    let x = DimDyn::from(x.slice());
    let y = DimDyn::from(y.slice());

    if x.len() < y.len() {
        if y.is_include(x) {
            false
        } else {
            panic!("This is bug please make issue");
        }
    } else if x.len() > y.len() {
        if x.is_include(y) {
            true
        } else {
            panic!("This is bug please make issue");
        }
    } else if x.is_include_bradcast(y) {
        true
    } else if y.is_include_bradcast(x) {
        false
    } else {
        panic!("This is bug please make issue");
    }
}

pub fn larger_shape<D1: DimTrait, D2: DimTrait>(x: D1, y: D2) -> DimDyn {
    let x = DimDyn::from(x.slice());
    let y = DimDyn::from(y.slice());

    if larger_shape_is_x(x, y) {
        x
    } else {
        y
    }
}

pub(crate) fn smaller_shape<D1: DimTrait, D2: DimTrait>(x: D1, y: D2) -> DimDyn {
    let x = DimDyn::from(x.slice());
    let y = DimDyn::from(y.slice());

    if larger_shape_is_x(x, y) {
        y
    } else {
        x
    }
}

impl DimDyn {
    pub fn new(dim: &[usize]) -> Self {
        if dim.len() > 6 {
            panic!("Dim must be smaller than 4");
        }
        let mut dim_dyn = DimDyn::default();
        for i in dim {
            dim_dyn.push_dim(*i)
        }
        dim_dyn
    }

    pub fn dim(&self) -> [usize; 6] {
        self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    pub fn get_len(&self) -> usize {
        self.len
    }

    pub(crate) fn inc_len(&mut self) {
        self.len += 1;
    }

    pub(crate) fn push_dim(&mut self, dim: usize) {
        self.dim[self.len] = dim;
        self.inc_len();
    }

    // otherのshapeがselfのshapeに含まれているかどうか
    // ex
    // self: [2, 3, 4]
    // other: [3, 4]
    // => true
    // other: [3, 4, 5]
    // => false
    pub fn is_include(&self, other: DimDyn) -> bool {
        // selfのshapeの後ろからotherのshapeを比較していく
        if self.len() < other.len() {
            return other.is_include(*self);
        }
        for i in 0..other.len {
            if self.dim[self.len - 1 - i] != other.dim[other.len - 1 - i] {
                return false;
            }
        }
        true
    }

    // selfとotherがadd, sub, mul, divで演算可能かを調べる
    // [10, 10, 1, 10]と[10, 1, 1, 10]は演算可能である
    // is_includeではfalseになるが演算可能なものを調べる
    pub(crate) fn is_include_bradcast(&self, other: DimDyn) -> bool {
        if self.len() < other.len() {
            panic!("this is bug please make issue");
        }
        for i in 0..other.len() {
            if self.dim[self.len() - 1 - i] == other.dim[other.len() - 1 - i]
                || other.dim[other.len() - i - 1] == 1
            {
                continue;
            } else {
                return false;
            }
        }
        true
    }
}

impl LessDimTrait for DimDyn {
    type LessDim = DimDyn;
}

impl GreaterDimTrait for DimDyn {
    type GreaterDim = DimDyn;
}

pub(crate) fn into_dyn<Din: DimTrait>(dim: Din) -> DimDyn {
    let mut dim_out = DimDyn::default();
    for i in 0..dim.len() {
        dim_out.push_dim(dim[i]);
    }
    dim_out
}

impl Index<usize> for DimDyn {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len {
            panic!("Index out of range");
        }
        &self.dim[index]
    }
}

impl IndexMut<usize> for DimDyn {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= self.len {
            panic!("Index out of range");
        }
        &mut self.dim[index]
    }
}

macro_rules! impl_range_index {
    ($trait:ident, $($ty:tt)*) => {
        impl Index<$trait$($ty)*> for DimDyn {
            type Output = [usize];

            fn index(&self, index: $trait$($ty)*) -> &Self::Output {
                &self.dim[index] as &[usize]
            }
        }

        // impl IndexMut<$trait$($ty)*> for DimDyn {
        //     fn index_mut(&mut self, index: $trait$($ty)*) -> &mut Self::Output {
        //         &mut DimDyn::from(&self.dim[index] as &[usize])
        //     }
        // }
    };
}

impl_range_index!(Range, <usize>);
impl_range_index!(RangeFrom, <usize>);
impl_range_index!(RangeFull,);
impl_range_index!(RangeInclusive, <usize>);
impl_range_index!(RangeTo, <usize>);
impl_range_index!(RangeToInclusive, <usize>);

impl IntoIterator for DimDyn {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    #[allow(clippy::unnecessary_to_owned)]
    fn into_iter(self) -> Self::IntoIter {
        self.dim[0..self.len()].to_vec().into_iter()
    }
}

impl DimTrait for DimDyn {
    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn is_overflow<D: DimTrait>(&self, index: D) -> bool {
        index.len() != self.len
    }

    fn slice(&self) -> &[usize] {
        if self.len == 0 {
            return &[];
        }
        &self.dim[0..self.len]
    }
}

impl From<&[usize]> for DimDyn {
    fn from(slice: &[usize]) -> Self {
        let mut dim_dyn = DimDyn::default();
        for i in slice {
            dim_dyn.push_dim(*i);
        }
        dim_dyn
    }
}

impl From<&DimDyn> for DimDyn {
    fn from(dim: &DimDyn) -> Self {
        *dim
    }
}

macro_rules! impl_from_slice_dim {
    ($name:ident, $number_of_elm:expr) => {
        impl From<&[usize; $number_of_elm]> for $name {
            fn from(slice: &[usize; $number_of_elm]) -> Self {
                let mut dim_dyn = $name::default();
                for i in 0..slice.len() {
                    dim_dyn.push_dim(slice[i]);
                }
                dim_dyn
            }
        }

        impl From<[usize; $number_of_elm]> for $name {
            fn from(slice: [usize; $number_of_elm]) -> Self {
                let mut dim_dyn = $name::default();
                for i in 0..slice.len() {
                    dim_dyn.push_dim(slice[i]);
                }
                dim_dyn
            }
        }
    };
    () => {};
}
impl_from_slice_dim!(DimDyn, 6);
impl_from_slice_dim!(DimDyn, 5);
impl_from_slice_dim!(DimDyn, 4);
impl_from_slice_dim!(DimDyn, 3);
impl_from_slice_dim!(DimDyn, 2);
impl_from_slice_dim!(DimDyn, 1);
impl_from_slice_dim!(DimDyn, 0);

#[cfg(test)]
mod dim_dyn {
    #[test]
    fn is_include_bradcast_2x4x5x5_1x4x1x1() {
        let x = super::DimDyn::new(&[2, 4, 5, 5]);
        let y = super::DimDyn::new(&[1, 4, 1, 1]);
        assert_eq!(x.is_include_bradcast(y), true);
    }
}

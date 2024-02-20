use std::ops::{Index, IndexMut};

use super::DimTrait;

#[derive(Clone, Debug, Default, PartialEq, Copy)]
pub struct DimDyn {
    dim: [usize; 4],
    len: usize,
}

impl DimDyn {
    pub fn new(dim: &[usize]) -> Self {
        if dim.len() > 4 {
            panic!("Dim must be smaller than 4");
        }
        let mut dim_dyn = DimDyn::default();
        for i in dim {
            dim_dyn.push_dim(*i)
        }
        dim_dyn
    }

    pub fn dim(&self) -> [usize; 4] {
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
    pub fn is_include(&self, other: &DimDyn) -> bool {
        if self.len < other.len {
            return false;
        }

        // selfのshapeの後ろからotherのshapeを比較していく
        for i in 0..other.len {
            if self.dim[self.len - 1 - i] != other.dim[other.len - 1 - i] {
                return false;
            }
        }
        true
    }
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
impl_from_slice_dim!(DimDyn, 4);
impl_from_slice_dim!(DimDyn, 3);
impl_from_slice_dim!(DimDyn, 2);
impl_from_slice_dim!(DimDyn, 1);

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
        for i in 0..dim.len() {
            dim_dyn[i] = dim[i];
        }
        dim_dyn.len = dim.len();
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
}

pub(crate) fn convert_dim<Din: DimTrait, Dout: DimTrait>(dim: Din) -> Dout {
    let mut dim_out = Dout::default();
    for i in 0..dim.len() {
        dim_out[i] = dim[i];
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
        self.dim.to_vec().into_iter()
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
}

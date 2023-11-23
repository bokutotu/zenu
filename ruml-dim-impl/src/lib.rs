use std::ops::Index;

use ruml_matrix_traits::dim::DimTrait;

#[derive(Clone, Debug, Copy)]
pub struct Dim0 {}

impl Dim0 {
    pub fn new() -> Self {
        Self {}
    }
}

impl Index<usize> for Dim0 {
    type Output = usize;

    fn index(&self, _: usize) -> &Self::Output {
        &0
    }
}

impl PartialEq for Dim0 {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Iterator for Dim0 {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl DimTrait for Dim0 {
    fn len(&self) -> usize {
        0
    }

    fn is_overflow<D: DimTrait>(&self, index: D) -> bool {
        if index.len() == 0 {
            false
        } else {
            true
        }
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Dim1 {
    dim: usize,
}

impl Dim1 {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Index<usize> for Dim1 {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        if index == 0 {
            &self.dim
        } else {
            panic!("Index out of range");
        }
    }
}

impl PartialEq for Dim1 {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl Iterator for Dim1 {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl DimTrait for Dim1 {
    fn len(&self) -> usize {
        1
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Dim2 {
    dim: [usize; 2],
}

impl Dim2 {
    pub fn new(dim: [usize; 2]) -> Self {
        Self { dim }
    }

    pub fn dim(&self) -> [usize; 2] {
        self.dim
    }
}

impl Index<usize> for Dim2 {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dim[index]
    }
}

impl IntoIterator for Dim2 {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.dim.to_vec().into_iter()
    }
}

impl PartialEq for Dim2 {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl DimTrait for Dim2 {
    fn len(&self) -> usize {
        2
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Dim3 {
    dim: [usize; 3],
}

impl Dim3 {
    pub fn new(dim: [usize; 3]) -> Self {
        Self { dim }
    }

    pub fn dim(&self) -> [usize; 3] {
        self.dim
    }
}

impl Index<usize> for Dim3 {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dim[index]
    }
}

impl IntoIterator for Dim3 {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.dim.to_vec().into_iter()
    }
}

impl PartialEq for Dim3 {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl DimTrait for Dim3 {
    fn len(&self) -> usize {
        3
    }
}

#[derive(Clone, Debug, Copy)]
pub struct Dim4 {
    dim: [usize; 4],
}

impl Dim4 {
    pub fn new(dim: [usize; 4]) -> Self {
        Self { dim }
    }

    pub fn dim(&self) -> [usize; 4] {
        self.dim
    }
}

impl Index<usize> for Dim4 {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dim[index]
    }
}

impl IntoIterator for Dim4 {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.dim.to_vec().into_iter()
    }
}

impl PartialEq for Dim4 {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl DimTrait for Dim4 {
    fn len(&self) -> usize {
        4
    }
}

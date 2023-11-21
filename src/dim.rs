use std::ops::{Index, IndexMut};

pub trait DimTrait:
    Index<usize> + Iterator<Item = usize> + Clone + Default + IndexMut<usize>
{
    fn len(&self) -> usize;
    fn is_overflow<D: DimTrait>(&self, index: D) -> bool;
}

#[derive(Clone, Default)]
pub struct Dim {
    data: Vec<usize>,
}

impl Dim {
    pub fn new(data: Vec<usize>) -> Self {
        Self { data }
    }

    pub fn data(&self) -> Vec<usize> {
        self.data.clone()
    }
}

impl Index<usize> for Dim {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Dim {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Iterator for Dim {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.data.pop()
    }
}

impl DimTrait for Dim {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_overflow<D: DimTrait>(&self, index: D) -> bool {
        if self.len() != index.len() {
            panic!("Dimension mismatch");
        }
        self.data().iter().zip(index.clone()).any(|(x, y)| *x <= y)
    }
}

pub fn cal_offset<D1: DimTrait, D2: DimTrait>(shape: D1, stride: D2) -> usize {
    if shape.len() != stride.len() {
        panic!("Dimension mismatch");
    }
    shape.clone().zip(stride.clone()).map(|(x, y)| x * y).sum()
}

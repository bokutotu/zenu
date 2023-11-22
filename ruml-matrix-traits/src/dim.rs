use std::ops::Index;

pub trait DimTrait:
    Index<usize, Output = isize> + IntoIterator<Item = isize> + Clone + Copy + PartialEq
{
    fn len(&self) -> usize;
    fn is_overflow<D: DimTrait>(&self, index: D) -> bool {
        if self.len() < index.len() {
            panic!("Dimension mismatch");
        }

        index.into_iter().zip(self.clone()).any(|(x, y)| x >= y)
    }
}

pub fn cal_offset<D1: DimTrait, D2: DimTrait>(shape: D1, stride: D2) -> isize {
    if shape.len() != stride.len() {
        panic!("Dimension mismatch");
    }
    shape
        .clone()
        .into_iter()
        .zip(stride.clone())
        .map(|(x, y)| x * y)
        .sum()
}

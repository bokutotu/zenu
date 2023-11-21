use std::ops::{Index, IndexMut};

pub trait Memory: Index<usize> + IndexMut<usize> {
    type Item: Default + Clone + Copy;

    fn allocate(size: usize) -> Self;
    fn as_ptr(&self) -> *const Self::Item;
    fn len(&self) -> usize;
}

pub struct CpuMemory<T> {
    pub data: Vec<T>,
}

impl<T> Index<usize> for CpuMemory<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for CpuMemory<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Default + Clone + Copy> Memory for CpuMemory<T> {
    type Item = T;

    fn allocate(size: usize) -> Self {
        Self {
            data: vec![Default::default(); size],
        }
    }

    fn as_ptr(&self) -> *const Self::Item {
        self.data.as_ptr()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

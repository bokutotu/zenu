use std::sync::Arc;

use ruml_matrix_traits::{
    memory::{Memory, OwnedMemory, ViewMemory},
    num::Num,
};

#[derive(Clone)]
pub struct CpuOwnedMemory<T: Num> {
    buffer: Vec<T>,
}

impl<T: Num> CpuOwnedMemory<T> {
    pub fn new(size: usize) -> Self {
        Self {
            buffer: vec![T::default(); size],
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.buffer
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buffer
    }
}

impl<T: Num> Memory for CpuOwnedMemory<T> {
    type Item = T;

    fn as_ptr(&self) -> *const Self::Item {
        self.buffer.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Item {
        self.buffer.as_mut_ptr()
    }
}

impl<T: Num> OwnedMemory for CpuOwnedMemory<T> {
    type View = CpuViewMemory<T>;

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn is_empty(&self) -> bool {
        todo!();
    }

    fn allocate(size: usize) -> Self {
        Self::new(size)
    }

    fn into_view(self, offset: usize) -> Self::View {
        CpuViewMemory::new(Arc::new(self), offset)
    }
}

#[derive(Clone)]
pub struct CpuViewMemory<T: Num> {
    buffer: Arc<CpuOwnedMemory<T>>,
    offset: usize,
}

impl<T: Num> CpuViewMemory<T> {
    pub fn new(buffer: Arc<CpuOwnedMemory<T>>, offset: usize) -> Self {
        Self { buffer, offset }
    }
}

impl<T: Num> Memory for CpuViewMemory<T> {
    type Item = T;

    fn as_ptr(&self) -> *const Self::Item {
        unsafe { self.buffer.as_ptr().add(self.offset) }
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Item {
        self.as_ptr() as *mut _
    }
}

impl<T: Num> ViewMemory for CpuViewMemory<T> {
    fn offset(&self) -> usize {
        self.offset
    }
}

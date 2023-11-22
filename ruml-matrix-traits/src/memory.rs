use crate::num::Num;

pub trait Memory {
    type Item: Num;

    fn as_ptr(&self) -> *const Self::Item;
    fn as_mut_ptr(&mut self) -> *mut Self::Item;
}

pub trait OwnedMemory: Memory {
    type View: ViewMemory;

    fn len(&self) -> usize;
    fn allocate(size: usize) -> Self;
    fn into_view(self, offset: usize) -> Self::View;
}

pub trait ViewMemory: Memory {
    fn offset(&self) -> usize;
}

use crate::num::Num;

pub trait Memory {
    type Item: Num;

    fn as_ptr(&self) -> *const Self::Item;
    fn as_mut_ptr(&mut self) -> *mut Self::Item;
    fn ptr_add(&self, offset: usize) -> &Self::Item {
        unsafe { &*self.as_ptr().add(offset) }
    }
}

pub trait OwnedMemory<'a>: Memory {
    type View: ViewMemory<'a>;

    fn from_vec(vec: Vec<Self::Item>) -> Self;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn allocate(size: usize) -> Self;
    fn to_view(&'a self, offset: usize) -> Self::View;
}

pub trait ViewMemory<'b>: Memory {
    type Owned: OwnedMemory<'b>;
    fn offset(&self) -> usize;
    fn to_owned(&self) -> Self::Owned;
}

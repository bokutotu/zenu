use crate::num::Num;

pub trait Memory {
    type Item: Num;

    fn from_vec(vec: Vec<Self::Item>) -> Self;
    fn as_ptr(&self) -> *const Self::Item;
    fn as_mut_ptr(&mut self) -> *mut Self::Item;
    fn ptr_add(&self, offset: usize) -> &Self::Item {
        unsafe {
            println!("{:?}", *(self.as_ptr() as *const Self::Item));
        }
        unsafe { &*self.as_ptr().add(offset) }
    }
}

pub trait OwnedMemory: Memory {
    type View: ViewMemory;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn allocate(size: usize) -> Self;
    fn to_view(&self, offset: usize) -> Self::View;
}

pub trait ViewMemory: Memory {
    type Owned: OwnedMemory;
    fn offset(&self) -> usize;
    fn to_owned(&self) -> Self::Owned;
}

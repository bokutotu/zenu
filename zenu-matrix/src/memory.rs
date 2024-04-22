use std::ptr::NonNull;

use crate::{
    blas::Blas,
    // element_wise::ElementWise,
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
};

pub trait MemAcc: Copy {
    type Item: Num;

    fn value(&self, ptr: NonNull<Self::Item>, offset: usize) -> Self::Item;
    fn set_value(&mut self, ptr: NonNull<Self::Item>, offset: usize, value: Self::Item);
    fn clone_ptr(&self, ptr: NonNull<Self::Item>, len: usize) -> NonNull<Self::Item>;
    fn drop(&self, ptr: *const Self::Item, len: usize);
    fn offset_ptr(&self, ptr: NonNull<Self::Item>, offset: usize) -> NonNull<Self::Item>;
}

/// Matrixの要素を保持するメモリを表すトレイト
#[allow(clippy::len_without_is_empty)]
pub trait Memory {
    type Item: Num;
    type Blas: Blas<Self::Item>;
    // type ElmentWise: ElementWise<Self::Item>;

    fn len(&self) -> usize;
    /// 確保しているメモリの先頭のポインタを返す
    /// offsetがある場合でもoffsetは考慮されない
    fn as_ptr(&self) -> *const Self::Item;
    fn as_ptr_offset(&self, offset: usize) -> *const Self::Item;
    fn value_offset(&self, offset: usize) -> Self::Item;
    fn get_offset(&self) -> usize;
    fn set_offset(&mut self, offset: usize);
}

pub trait ToViewMemory: Memory {
    fn to_view(&self, offset: usize) -> ViewMem<Self::Item>;
}

pub trait ToViewMutMemory: Memory {
    fn to_view_mut(&mut self, offset: usize) -> ViewMutMem<Self::Item>;
}

pub trait ToOwnedMemory: Memory {
    type Owned: Owned<Item = Self::Item>;

    fn to_owned_memory(&self) -> Self::Owned;
}

/// Memoryの中でも値を保持するメモリを表すトレイト
pub trait Owned: Memory + ToViewMemory + ToViewMutMemory + Clone + ToOwnedMemory + 'static {
    fn from_vec(vec: Vec<Self::Item>) -> Self;
}

/// Memoryの中でも値を保持するメモリを表すトレイト
pub trait View: Memory + ToOwnedMemory + Clone + ToViewMemory {}

/// Memoryの中でも値を保持するメモリを表すトレイト(可変参照)
pub trait ViewMut: Memory + ToOwnedMemory + ToViewMemory + ToViewMutMemory {
    fn as_mut_ptr(&self) -> *mut Self::Item;
    fn as_mut_ptr_offset(&self, offset: usize) -> *mut Self::Item {
        unsafe { self.as_mut_ptr().add(self.get_offset() + offset) }
    }
}

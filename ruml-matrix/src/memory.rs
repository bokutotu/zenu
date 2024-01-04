use crate::{blas::Blas, element_wise::ElementWise, num::Num};

/// Matrixの要素を保持するメモリを表すトレイト
#[allow(clippy::len_without_is_empty)]
pub trait Memory {
    type Item: Num;
    type Blas: Blas<Self::Item>;
    type ElmentWise: ElementWise<Self::Item>;

    fn len(&self) -> usize;
    /// 確保しているメモリの先頭のポインタを返す
    /// offsetがある場合でもoffsetは考慮されない
    fn as_ptr(&self) -> *const Self::Item;
    fn as_ptr_offset(&self, offset: usize) -> *const Self::Item {
        if self.get_offset() + offset >= self.len() {
            panic!("out of range");
        }
        unsafe { self.as_ptr().add(self.get_offset() + offset) }
    }
    fn value_offset(&self, offset: usize) -> Self::Item {
        unsafe { *self.as_ptr_offset(offset) }
    }
    fn get_offset(&self) -> usize;
    fn set_offset(&mut self, offset: usize);
}

pub trait ToViewMemory: Memory {
    type View<'a>: ViewMemory<Item = Self::Item>
    where
        Self: 'a;

    fn to_view(&self, offset: usize) -> Self::View<'_>;
}

pub trait ToViewMutMemory: Memory {
    type ViewMut<'a>: ViewMutMemory<Item = Self::Item>
    where
        Self: 'a;

    fn to_view_mut(&mut self, offset: usize) -> Self::ViewMut<'_>;
}

pub trait ToOwnedMemory: Memory {
    type Owned: OwnedMemory<Item = Self::Item>;

    fn to_owned_memory(&self) -> Self::Owned;
}

/// Memoryの中でも値を保持するメモリを表すトレイト
pub trait OwnedMemory: Memory + ToViewMemory + ToViewMutMemory + Clone {
    fn from_vec(vec: Vec<Self::Item>) -> Self;
}

/// Memoryの中でも値を保持するメモリを表すトレイト
pub trait ViewMemory: Memory + ToOwnedMemory + Clone + ToViewMemory {}

/// Memoryの中でも値を保持するメモリを表すトレイト(可変参照)
pub trait ViewMutMemory: Memory + ToOwnedMemory + ToViewMemory + ToViewMutMemory {
    fn as_mut_ptr(&self) -> *mut Self::Item;
    fn as_mut_ptr_offset(&self, offset: usize) -> *mut Self::Item {
        unsafe { self.as_mut_ptr().add(self.get_offset() + offset) }
    }
}

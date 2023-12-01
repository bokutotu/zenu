use crate::num::Num;

/// Matrixの要素を保持するメモリを表すトレイト
pub trait Memory {
    type Item: Num;

    fn len(&self) -> usize;
    fn as_ptr(&self) -> *const Self::Item;
    fn ptr_offset(&self, offset: usize) -> Self::Item {
        unsafe { *self.as_ptr().add(self.get_offset() + offset) }
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
pub trait ViewMutMemory: Memory + ToOwnedMemory + ToViewMemory {}

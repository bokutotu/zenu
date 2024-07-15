//! bytesが固定のメモリプール
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    ops::{
        Bound::{Included, Unbounded},
        Deref,
    },
    rc::Rc,
};

use super::static_buffer::StaticSizeBuffer;
use crate::device::DeviceBase;

#[derive(Clone)]
pub struct RcBuffer<D: DeviceBase, const N: usize>(pub Rc<RefCell<StaticSizeBuffer<D, N>>>);

#[derive(Default)]
pub struct PtrBufferMap<D: DeviceBase, const N: usize>(pub HashMap<*mut u8, RcBuffer<D, N>>);

impl<D: DeviceBase, const N: usize> PtrBufferMap<D, N> {
    pub fn insert(&mut self, buffer: RcBuffer<D, N>) {
        let ptr = buffer.0.deref().borrow().ptr();
        self.0.insert(ptr, buffer);
    }

    pub fn remove(&mut self, ptr: *mut u8) -> Option<RcBuffer<D, N>> {
        self.0.remove(&ptr)
    }

    pub fn pop(&mut self) -> (*mut u8, RcBuffer<D, N>) {
        let (ptr, _) = { &self.0.iter().next().unwrap() };
        let ptr = **ptr;
        let buffer = self.0.remove(&ptr).unwrap();
        (ptr, buffer)
    }
}

/// 未使用バイト数とそのバイト数を持つバッファのマップ
/// 未使用バイト数が同じバッファを複数持つことがあるため, PtrBufferMapを使用する
///
#[derive(Default)]
pub struct UnusedBytesPtrBufferMap<D: DeviceBase, const N: usize>(
    pub BTreeMap<usize, PtrBufferMap<D, N>>,
);

impl<D: DeviceBase, const N: usize> UnusedBytesPtrBufferMap<D, N> {
    /// 要求されたメモリよりも多くの未使用メモリを持つバッファの中で最小の未使用バイト数を返す
    pub fn smallest_unused_bytes_over_request(&self, bytes: usize) -> Option<usize> {
        self.0
            .range((Included(&bytes), Unbounded))
            .next()
            .map(|(unused_bytes, _)| *unused_bytes)
    }

    fn _get_ptr_buffer_map(&mut self, unused_bytes: usize) -> Option<PtrBufferMap<D, N>> {
        self.0.remove(&unused_bytes)
    }

    /// 未使用バイト数がunused_bytesのバッファを取得
    /// unused_bytesに対応するPtrBufferMapが存在しない場合、Noneを返す
    /// 返却されるRcBufferは、Selfからは削除される
    pub fn pop_unused_bytes_ptr_buffer(&mut self, unused_bytes: usize) -> RcBuffer<D, N> {
        let mut ptr_buffer_map = self._get_ptr_buffer_map(unused_bytes).unwrap();
        let (_, buffer) = ptr_buffer_map.pop();
        if !ptr_buffer_map.0.is_empty() {
            self.0.insert(unused_bytes, ptr_buffer_map);
        }
        buffer
    }

    pub fn insert(&mut self, buffer: RcBuffer<D, N>) {
        let unused_bytes = buffer.0.deref().borrow().get_unused_bytes();
        // unused_bytesに対応するPtrBufferMapが存在しない場合、新たに作成する
        self.0.entry(unused_bytes).or_default().insert(buffer);
    }
}

/// 固定サイズのメモリプール
///
/// 2つのマップを使用する理由
/// - unused_bytes_ptr_buffer_map
///   要求バイト数を満たす最小のbufferを高速に取得するため(O(logN))
/// - alloced_ptr_buffer_map
///   ポインタを解放する際、確保されたptrはどのbufferに属しているかを高速に取得するため(O(1))
///
/// 1. 確保の流れ
///    1. 要求のバイト数よりも多くの未使用bytesを持つバッファがある場合
///       1. unused_bytes_ptr_buffer_mapから条件を満たす最小のunused_bytesのBufferを取得
///          (unused_bytes_ptr_buffer_mapからBufferは削除される)
///       2. Bufferからbytesを確保
///       3. alloced_ptr_buffer_mapにBufferのポインタとBufferを追加
///       4. unused_bytes_ptr_buffer_mapにBufferを追加
///       5. 確保したポインタを返す
///    2. 要求のバイト数よりも多くの未使用bytesを持つバッファがある場合
///       1. 新たにBufferを作成
///       2. Bufferからbytesを確保
///       3. alloced_ptr_buffer_mapにBufferのポインタとBufferを追加
///       4. unused_bytes_ptr_buffer_mapにBufferを追加
///       5. 確保したポインタを返す
/// 2. 解放の流れ
///    1. alloced_ptr_buffer_mapからポインタに対応するBufferを取得
///       (alloced_ptr_buffer_mapからBufferは削除される)
///    2. unused_bytesを取得
///    3. unused_bytesからunused_bytes_ptr_buffer_mapからBufferを取得
///       (unused_bytes_ptr_buffer_mapからBufferは削除される)
///    2. Bufferからポインタを解放
///    3. unused_bytes_ptr_buffer_mapにBufferを追加
///
#[derive(Default)]
pub struct StaticMemPool<D: DeviceBase, const N: usize> {
    /// 未使用バイト数とそのバイト数を持つバッファのマップ
    pub unused_bytes_ptr_buffer_map: UnusedBytesPtrBufferMap<D, N>,
    /// 確保されたポインタとそのポインタを保持するバッファのマップ
    pub alloced_ptr_buffer_map: HashMap<*mut u8, RcBuffer<D, N>>,
}

impl<D: DeviceBase, const N: usize> StaticMemPool<D, N> {
    pub fn smallest_unused_bytes_over_request(&self, bytes: usize) -> Option<usize> {
        self.unused_bytes_ptr_buffer_map
            .smallest_unused_bytes_over_request(bytes)
    }

    pub fn try_alloc(&mut self, bytes: usize) -> Result<*mut u8, ()> {
        if let Some(unused_bytes) = self
            .unused_bytes_ptr_buffer_map
            .smallest_unused_bytes_over_request(bytes)
        {
            let buffer = self
                .unused_bytes_ptr_buffer_map
                .pop_unused_bytes_ptr_buffer(unused_bytes);
            let ptr = buffer.0.deref().borrow_mut().try_alloc(bytes)?;
            self.alloced_ptr_buffer_map.insert(ptr, buffer.clone());
            self.unused_bytes_ptr_buffer_map.insert(buffer);
            Ok(ptr)
        } else {
            let buffer = RcBuffer(Rc::new(RefCell::new(StaticSizeBuffer::new()?)));
            let ptr = buffer.0.deref().borrow_mut().try_alloc(bytes)?;
            self.alloced_ptr_buffer_map.insert(ptr, buffer.clone());
            self.unused_bytes_ptr_buffer_map.insert(buffer);
            Ok(ptr)
        }
    }

    pub fn free(&mut self, ptr: *mut u8) -> Result<(), ()> {
        let buffer = self.alloced_ptr_buffer_map.remove(&ptr).ok_or(())?;

        let unused_bytes = buffer.0.deref().borrow().get_unused_bytes();
        let _ = self
            .unused_bytes_ptr_buffer_map
            .pop_unused_bytes_ptr_buffer(unused_bytes);
        buffer.0.deref().borrow_mut().try_free(ptr)?;
        self.unused_bytes_ptr_buffer_map.insert(buffer);
        Ok(())
    }
}

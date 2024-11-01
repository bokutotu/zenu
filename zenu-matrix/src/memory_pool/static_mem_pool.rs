//! bytesが固定のメモリプール
use std::{
    collections::{BTreeMap, HashMap},
    ops::Bound::{Included, Unbounded},
    sync::{Arc, Mutex},
};

use super::{static_buffer::StaticSizeBuffer, MemPoolError};
use crate::device::DeviceBase;

#[derive(Clone)]
pub struct ArcBuffer<D: DeviceBase, const N: usize>(pub Arc<Mutex<StaticSizeBuffer<D, N>>>);

impl<D: DeviceBase, const N: usize> ArcBuffer<D, N> {
    fn ptr(&self) -> *mut u8 {
        self.0.lock().unwrap().ptr()
    }
    fn try_free(&self, ptr: *mut u8) -> Result<(), MemPoolError> {
        self.0.lock().unwrap().try_free(ptr)
    }
    fn get_unused_bytes(&self) -> usize {
        self.0.lock().unwrap().get_unused_bytes()
    }
}

#[derive(Default)]
pub struct PtrBufferMap<D: DeviceBase, const N: usize>(pub HashMap<*mut u8, ArcBuffer<D, N>>);

impl<D: DeviceBase, const N: usize> PtrBufferMap<D, N> {
    pub fn insert(&mut self, buffer: ArcBuffer<D, N>) {
        let ptr = buffer.0.lock().unwrap().ptr();
        self.0.insert(ptr, buffer);
    }

    fn remove(&mut self, ptr: *mut u8) -> Option<ArcBuffer<D, N>> {
        self.0.remove(&ptr)
    }

    pub fn pop(&mut self) -> (*mut u8, ArcBuffer<D, N>) {
        let (ptr, _) = { &self.0.iter().next().unwrap() };
        let ptr = **ptr;
        let buffer = self.remove(ptr).unwrap();
        (ptr, buffer)
    }
}

/// 未使用バイト数とそのバイト数を持つバッファのマップ
/// 未使用バイト数が同じバッファを複数持つことがあるため, `PtrBufferMap`を使用する
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

    fn pop_ptr_buffer_map(&mut self, unused_bytes: usize) -> Option<PtrBufferMap<D, N>> {
        self.0.remove(&unused_bytes)
    }

    /// 未使用バイト数が`unused_bytes`のバッファを取得
    /// `unused_bytes`に対応する `PtrBufferMap`が存在しない場合、`None`を返す
    /// 返却される`ArcBuffer`は、Selfからは削除される
    pub fn pop_unused_bytes_ptr_buffer(&mut self, unused_bytes: usize) -> ArcBuffer<D, N> {
        let mut ptr_buffer_map = self.pop_ptr_buffer_map(unused_bytes).unwrap();
        let (_, buffer) = ptr_buffer_map.pop();
        if !ptr_buffer_map.0.is_empty() {
            self.0.insert(unused_bytes, ptr_buffer_map);
        }
        buffer
    }

    /// 未使用バイト数と先頭ポインタを持つバッファを`UnusedBytesPtrBufferMap`から削除
    pub fn remove(&mut self, unused_bytes: usize, ptr: *mut u8) {
        let map = self.0.get_mut(&unused_bytes).unwrap();
        map.remove(ptr);
        if map.0.is_empty() {
            self.0.remove(&unused_bytes);
        }
    }

    pub fn insert(&mut self, buffer: ArcBuffer<D, N>) {
        let unused_bytes = buffer.0.lock().unwrap().get_unused_bytes();
        // unused_bytesに対応するPtrBufferMapが存在しない場合、新たに作成する
        self.0.entry(unused_bytes).or_default().insert(buffer);
    }
}

/// 固定サイズのメモリプール
///
/// 2つのマップを使用する理由
/// - `unused_bytes_ptr_buffer_map`
///   要求バイト数を満たす最小のbufferを高速に取得するため(O(logN))
/// - `alloced_ptr_buffer_map`
///   ポインタを解放する際、確保されたptrはどのbufferに属しているかを高速に取得するため(O(1))
///
/// 1. 確保の流れ
///    1. 要求のバイト数よりも多くの未使用bytesを持つバッファがある場合
///       1. `unused_bytes_ptr_buffer_map`から条件を満たす最小の`unused_bytes`の`Buffer`を取得
///          (`unused_bytes_ptr_buffer_map`から`Buffer`は削除される)
///       2. `Buffer`から`bytes`を確保
///       3. `alloced_ptr_buffer_map`に`Buffer`のポインタとBufferを追加
///       4. `unused_bytes_ptr_buffer_map`に`Buffer`を追加
///       5. 確保したポインタを返す
///    2. 要求のバイト数よりも多くの未使用bytesを持つバッファがある場合
///       1. 新たに`Buffer`を作成
///       2. `Buffer`から`bytes`を確保
///       3. `alloced_ptr_buffer_map`に`Buffer`のポインタと`Buffer`を追加
///       4. `unused_bytes_ptr_buffer_map`に`Buffer`を追加
///       5. 確保したポインタを返す
/// 2. 解放の流れ
///    1. `alloced_ptr_buffer_map`からポインタに対応する`Buffer`を取得
///       (`alloced_ptr_buffer_map`から`Buffer`は削除される)
///    2. `unused_bytes`を取得
///    3. `unused_bytes`から`unused_bytes_ptr_buffer_map`から`Buffer`を取得
///       (`unused_bytes_ptr_buffer_map`から`Buffer`は削除される)
///    2. `Buffer`からポインタを解放
///    3. `unused_bytes_ptr_buffer_map`に`Buffer`を追加
///
#[derive(Default)]
pub struct StaticMemPool<D: DeviceBase, const N: usize> {
    /// 未使用バイト数とそのバイト数を持つバッファのマップ
    pub unused_bytes_ptr_buffer_map: UnusedBytesPtrBufferMap<D, N>,
    /// 確保されたポインタとそのポインタを保持するバッファのマップ
    pub alloced_ptr_buffer_map: HashMap<*mut u8, ArcBuffer<D, N>>,
}

impl<D: DeviceBase, const N: usize> StaticMemPool<D, N> {
    fn smallest_unused_bytes_over_request(&self, bytes: usize) -> Option<usize> {
        self.unused_bytes_ptr_buffer_map
            .smallest_unused_bytes_over_request(bytes)
    }

    pub fn try_alloc(&mut self, bytes: usize) -> Result<*mut u8, MemPoolError> {
        if let Some(unused_bytes) = self.smallest_unused_bytes_over_request(bytes) {
            let buffer = self
                .unused_bytes_ptr_buffer_map
                .pop_unused_bytes_ptr_buffer(unused_bytes);
            let ptr = buffer.0.lock().unwrap().try_alloc(bytes)?;
            self.alloced_ptr_buffer_map.insert(ptr, buffer.clone());
            self.unused_bytes_ptr_buffer_map.insert(buffer);
            Ok(ptr)
        } else {
            let buffer = ArcBuffer(Arc::new(Mutex::new(StaticSizeBuffer::new()?)));
            let ptr = buffer.0.lock().unwrap().try_alloc(bytes)?;
            self.alloced_ptr_buffer_map.insert(ptr, buffer.clone());
            self.unused_bytes_ptr_buffer_map.insert(buffer);
            Ok(ptr)
        }
    }

    pub fn try_free(&mut self, ptr: *mut u8) -> Result<(), MemPoolError> {
        let buffer = self
            .alloced_ptr_buffer_map
            .remove(&ptr)
            .ok_or(MemPoolError::StaticMemPoolFreeError)?;
        let start_ptr = buffer.ptr();
        let unused_bytes = buffer.get_unused_bytes();
        self.unused_bytes_ptr_buffer_map
            .remove(unused_bytes, start_ptr);
        buffer.try_free(ptr)?;
        self.unused_bytes_ptr_buffer_map.insert(buffer);
        Ok(())
    }

    pub fn contains(&self, ptr: *mut u8) -> bool {
        self.alloced_ptr_buffer_map.contains_key(&ptr)
    }
}

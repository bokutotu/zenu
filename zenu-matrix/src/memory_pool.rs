//!
//! メモリプールの実装
//! メモリプールはメモリの確保の効率かのために使用する
//! メモリプールは1MB以上と以下で固定サイズのプールを使用する。
//! 1MB以下のメモリは2MBのプールを使用する。順次、プールを分割する
//! 1MB以上のメモリは20MBのプールを分割して使用する。
//! 20MB以上のメモリは、動的サイズのプールを作成する。
//! 20MB以上のプールはフラグメンテーションを防ぐために、分割は行わない。
//! メモリ確保は以下の手順で行う。
//! 1. メモリプールに空きメモリがあるか確認する
//! 2. 空きメモリがある場合、確保済みのメモリを返す
//! 3. 空きメモリがない場合、新たにメモリを確保する
//!    3.1 メモリプールに空きメモリがあるか確認する
//!        1MB以下のメモリは2MBのプールを使用する。順次、プールを分割する
//!        1MB以上のメモリは20MBのプールを分割して使用する。
//!        20MB以上のメモリは、動的サイズのプールを作成する。
//!    3.2 空きメモリがない場合
//!        一番大きい未使用メモリを解放し、もう一度メモリの確保を試みる
//!        失敗した場合、全ての秋メモリを解放し、メモリの確保を試みる
//!        失敗した場合、メモリの確保に失敗する
//!
//! ワークフローは以下
//! [開始] -> [メモリ割り当て要求]
//!            |
//!            v
//! [キャッシュブロックを検索] --(適切なブロックあり)--> [ブロックを返す]
//!            |                                                |
//!            | (適切なブロックなし)                           |
//!            v                                                |
//! [cudaMalloc実行] --(成功)--> [ブロックを返す]               |
//!            |                                                |
//!            | (失敗)                                         |
//!            v                                                |
//! [非分割キャッシュブロック1つ解放] --> [再度cudaMalloc]      |
//!            |                         |                      |
//!            |                         | (失敗)               |
//!            |                         v                      |
//!            |    [全非分割キャッシュブロック解放] ---------> |
//!            |                         |                      |
//!            |                         v                      |
//!            |                   [再度cudaMalloc]             |
//!            |                         |                      |
//!            |                         v                      |
//!            |              (成功) [ブロックを返す]           |
//!            |                         |                      |
//!            v                         |                      |
//! [アロケーション失敗]                 |                      |
//!            |                         |                      |
//!            |                         |                      |
//!            v                         v                      v
//!         [終了] <----------------------[ストリームに関連付け]
//!
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Deref,
    rc::Rc,
};

use crate::device::Device;

/// 長さが固定のメモリプール
/// ptrは確保した先頭のポインタ
/// malloc_rangesは確保済みのポインタの先頭と末尾のポインタのセット
/// non_used_bytesは未使用のバイト数
/// _markerはデバイスの型を保持する
///
/// メモリが解放される時、解放されるメモリの先頭のポインタが渡される。
/// そのポインタがmalloc_rangesに含まれている場合、そのポインタを削除する。
/// 解放されるポインタがmalloc_rangesの最後尾の場合のみ、non_used_bytesを更新する。
struct StaticDataPtr<D: Device, const N: usize> {
    ptr: *mut u8,
    malloc_ranges: Rc<RefCell<BTreeMap<*mut u8, *mut u8>>>,
    non_used_bytes: Rc<RefCell<usize>>,
    _marker: std::marker::PhantomData<D>,
}

impl<D: Device, const N: usize> StaticDataPtr<D, N> {
    fn new() -> Self {
        let ptr = D::alloc(N);
        Self {
            ptr,
            malloc_ranges: Rc::new(RefCell::new(BTreeMap::new())),
            non_used_bytes: Rc::new(RefCell::new(N)),
            _marker: std::marker::PhantomData,
        }
    }

    fn insert_alloc_range(&self, start: *mut u8, bytes: usize) {
        self.malloc_ranges
            .borrow_mut()
            .insert(start, unsafe { start.add(bytes) });
    }

    fn decrease_non_used_bytes(&self, bytes: usize) {
        *self.non_used_bytes.borrow_mut() -= bytes;
    }

    fn increase_non_used_bytes(&self, bytes: usize) {
        *self.non_used_bytes.borrow_mut() += bytes;
    }

    fn malloc_range_len(&self) -> usize {
        let map = self.malloc_ranges.deref();
        let map = map.borrow();
        map.len()
    }

    fn last_end_ptr(&self) -> *mut u8 {
        // DataPtrの中身が一歳使用されていない場合
        if self.malloc_range_len() == 0 {
            return self.ptr;
        }
        let mut map = self.malloc_ranges.borrow_mut();
        let ptr = map.last_entry().unwrap();
        *ptr.get()
    }

    /// StaticDataPtrのメモリからbytes分のメモリを確保する
    /// この関数は必ずbytesがnon_used_bytes以下であることを条件にしている
    fn alloc_ptr(&self, bytes: usize) -> *mut u8 {
        self.decrease_non_used_bytes(bytes);
        let last_ptr = self.last_end_ptr();

        // non_used_bytesを更新
        self.insert_alloc_range(last_ptr, bytes);

        last_ptr
    }

    fn free_ptr(&self, ptr: *mut u8) {
        let last_ptr = self.last_end_ptr();
        let malloced_end = self.malloc_ranges.borrow_mut().remove(&ptr).unwrap();
        if last_ptr == malloced_end {
            // ptr ~ malloced_endのbytes数
            let free_bytes = malloced_end as usize - ptr as usize;
            self.increase_non_used_bytes(free_bytes);
        }
    }
}

impl<D: Device, const N: usize> Drop for StaticDataPtr<D, N> {
    fn drop(&mut self) {
        if *(self.non_used_bytes).borrow() != N {
            panic!("non_used_bytes is not N");
        }
        if self.malloc_range_len() != 0 {
            panic!("malloc_ranges is not empty");
        }
        D::drop_ptr(self.ptr, N);
    }
}

impl<D: Device, const N: usize> Default for StaticDataPtr<D, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Device, const N: usize> Eq for StaticDataPtr<D, N> {}

impl<D: Device, const N: usize> PartialEq for StaticDataPtr<D, N> {
    fn eq(&self, other: &Self) -> bool {
        self.non_used_bytes == other.non_used_bytes
    }
}

impl<D: Device, const N: usize> PartialOrd for StaticDataPtr<D, N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<D: Device, const N: usize> Ord for StaticDataPtr<D, N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.non_used_bytes.cmp(&other.non_used_bytes)
    }
}

const SMALL_POOL_SIZE: usize = 2 * 1024 * 1024;

const BIG_POOL_SIZE: usize = 20 * 1024 * 1024;

#[derive(Default)]
pub struct StaticMemoryPool<D: Device> {
    small_pool: Rc<RefCell<BTreeSet<Rc<StaticDataPtr<D, SMALL_POOL_SIZE>>>>>,
    big_pool: Rc<RefCell<BTreeSet<Rc<StaticDataPtr<D, BIG_POOL_SIZE>>>>>,
    small_alloced_ptr: Rc<RefCell<HashMap<*mut u8, Rc<StaticDataPtr<D, SMALL_POOL_SIZE>>>>>,
    big_alloced_ptr: Rc<RefCell<HashMap<*mut u8, Rc<StaticDataPtr<D, BIG_POOL_SIZE>>>>>,
}

//// メモリプールの空きの中で、要求されたbytes分よりも大きく、かつ最小のStaticDataPtrを返す
fn find_min_non_used_bytes<D: Device, const N: usize>(
    pool: &Rc<RefCell<BTreeSet<Rc<StaticDataPtr<D, N>>>>>,
    bytes: usize,
) -> Option<Rc<StaticDataPtr<D, N>>>
where
    StaticDataPtr<D, N>: Eq + Ord,
{
    let pool = pool.borrow();
    let mut iter = pool.iter();
    let mut min_non_used_bytes = None;
    while let Some(ptr) = iter.next() {
        if ptr.non_used_bytes.borrow().deref() >= &bytes {
            min_non_used_bytes = Some(ptr.clone());
            break;
        }
    }
    min_non_used_bytes
}

impl StaticMemoryPool<crate::device::cpu::Cpu> {
    fn small_pool_len(&self) -> usize {
        (*self.small_pool).borrow().len()
    }

    fn big_pool_len(&self) -> usize {
        (*self.big_pool).borrow().len()
    }

    fn small_alloc(&self, bytes: usize) -> *mut u8 {
        todo!();
    }

    fn big_alloc(&self, bytes: usize) -> *mut u8 {
        todo!();
    }
}

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

use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::device::DeviceBase;

use self::{dynamic_pool::DynMemPool, static_mem_pool::StaticMemPool};
mod data_ptr;
mod dynamic_buffer;
mod dynamic_pool;
mod static_buffer;
mod static_mem_pool;
mod test;

// 2MB
pub const SMALL_BUFFER_SIZE: usize = 2 * 1024 * 1024;

// 20MB
pub const LARGE_BUFFER_SIZE: usize = 20 * 1024 * 1024;

// 10kb ブッファをきりはりする際のバッファの真ん中にある空き領域のサイズ
pub const MIDDLE_BUFFER_SIZE: usize = 10 * 1024;

#[derive(Default)]
pub struct MemPool<D: DeviceBase> {
    small_pool: Rc<RefCell<StaticMemPool<D, SMALL_BUFFER_SIZE>>>,
    large_pool: Rc<RefCell<StaticMemPool<D, LARGE_BUFFER_SIZE>>>,
    dynamic_pool: Rc<RefCell<DynMemPool<D>>>,
}

impl<D: DeviceBase> MemPool<D> {
    pub fn try_alloc(&self, bytes: usize) -> Result<*mut u8, ()> {
        if bytes >= LARGE_BUFFER_SIZE {
            self.dynamic_pool.deref().borrow_mut().try_alloc(bytes)
        } else if bytes >= SMALL_BUFFER_SIZE {
            self.large_pool.deref().borrow_mut().try_alloc(bytes)
        } else {
            self.small_pool.deref().borrow_mut().try_alloc(bytes)
        }
    }
}

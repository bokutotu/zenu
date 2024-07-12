use crate::device::DeviceBase;

use super::{data_ptr::DataPtr, MIDDLE_BUFFER_SIZE};
use std::collections::BTreeMap;

pub(super) struct StaticSizeBuffer<D: DeviceBase, const N: usize> {
    data: DataPtr<D>,
    // key is sttart address of used buffer
    // value is end address of used buffer
    used_buffer_range: BTreeMap<*mut u8, *mut u8>,
}

impl<D: DeviceBase, const N: usize> Default for StaticSizeBuffer<D, N> {
    fn default() -> Self {
        StaticSizeBuffer {
            data: DataPtr::new(N),
            // unused_bytes: N,
            used_buffer_range: BTreeMap::new(),
        }
    }
}

impl<D: DeviceBase, const N: usize> StaticSizeBuffer<D, N> {
    fn last_ptr(&mut self) -> Option<*mut u8> {
        self.used_buffer_range
            .last_entry()
            .map(|entry| *entry.get())
    }

    // 確保するメモリの始点と終点を返す
    fn start_end_ptr(&mut self, bytes: usize) -> (*mut u8, *mut u8) {
        self.last_ptr()
            .map(|end| {
                let start = unsafe { end.add(MIDDLE_BUFFER_SIZE) };
                let end = unsafe { start.add(bytes) };
                (start, end)
            })
            .unwrap_or_else(|| {
                let start = self.data.ptr;
                let end = unsafe { start.add(bytes) };
                (start, end)
            })
    }

    pub fn try_alloc(&mut self, bytes: usize) -> Result<*mut u8, ()> {
        if self.get_used_bytes() < bytes {
            return Err(());
        }
        let (start, end) = self.start_end_ptr(bytes);

        if end > unsafe { self.data.ptr.add(N) } {
            return Err(());
        }

        self.used_buffer_range.insert(start, end);
        Ok(start)
    }

    pub fn try_free(&mut self, ptr: *mut u8) -> Result<(), ()> {
        self.used_buffer_range.remove(&ptr).ok_or(())?;
        Ok(())
    }

    pub fn get_used_bytes(&mut self) -> usize {
        match self.last_ptr() {
            None => N,
            Some(last_ptr) => {
                let last = unsafe { last_ptr.add(MIDDLE_BUFFER_SIZE) };
                N - (last as usize - self.data.ptr as usize)
            }
        }
    }
}

#[cfg(test)]
mod static_buffer {
    use serde::Serialize;

    use crate::{
        device::DeviceBase,
        memory_pool::{static_buffer::StaticSizeBuffer, MIDDLE_BUFFER_SIZE},
        num::Num,
    };

    #[derive(Default, Clone, Copy, Serialize)]
    struct MockDeviceBase;

    impl DeviceBase for MockDeviceBase {
        fn zeros<T: Num>(_len: usize) -> *mut T {
            todo!();
        }
        fn alloc(_num_bytes: usize) -> *mut u8 {
            0 as *mut u8
        }
        fn drop_ptr<T>(_ptr: *mut T, _len: usize) {}
        fn get_item<T: Num>(_ptr: *const T, _offset: usize) -> T {
            todo!();
        }
        fn from_vec<T: Num>(_vec: Vec<T>) -> *mut T {
            todo!();
        }
        fn clone_ptr<T>(_ptr: *const T, _len: usize) -> *mut T {
            todo!();
        }
        fn assign_item<T: Num>(_ptr: *mut T, _offset: usize, _value: T) {
            todo!();
        }
    }

    #[test]
    fn too_large_alloc() {
        let mut buffer = StaticSizeBuffer::<MockDeviceBase, 1024>::default();
        let ptr = buffer.try_alloc(4000);
        assert_eq!(ptr, Err(()));
    }

    // 領域を2回確保し、アドレスを確認する
    #[test]
    fn alloc_twice() {
        let mut buffer = StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::default();
        let first_ptr = buffer.try_alloc(1024).unwrap();
        assert_eq!(first_ptr as usize, 0);
        let second_ptr = buffer.try_alloc(2048).unwrap();
        assert_eq!(second_ptr as usize, 1024 + MIDDLE_BUFFER_SIZE);
    }

    const BUF_LEN: usize = 2 * 1024 * 1024;
    // 3回領域を確保し,違う順番で解放する [1, 2, 3], [2, 3], [3, 2, 1],
    // 1つめは 1 << 15 byte, 2つめは 1 << 10byte, 3つめは 1 << 20byteのメモリを確保する
    fn alloc_3_fragments() -> (
        StaticSizeBuffer<MockDeviceBase, BUF_LEN>,
        *mut u8,
        *mut u8,
        *mut u8,
    ) {
        let mut buffer = StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::default();
        let ptr1 = buffer.try_alloc(1 << 15).unwrap();
        let ptr2 = buffer.try_alloc(1 << 10).unwrap();
        let ptr3 = buffer.try_alloc(1 << 20).unwrap();
        (buffer, ptr1, ptr2, ptr3)
    }

    #[test]
    fn alloc_3() {
        let (_, ptr1, ptr2, ptr3) = alloc_3_fragments();
        let ptr2_: usize = 1 << 15 + MIDDLE_BUFFER_SIZE;
        let ptr3_: usize = 1 << 15 + MIDDLE_BUFFER_SIZE + (1 << 10);
        assert_eq!(ptr1 as usize, 0);
        assert_eq!(ptr2 as usize, ptr2_);
        assert_eq!(ptr3 as usize, ptr3_);
    }
}

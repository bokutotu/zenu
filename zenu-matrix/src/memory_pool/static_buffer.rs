use super::data_ptr::DataPtr;

pub(super) struct StaticSizeBuffer<D: DeviceBase, const N: usize> {
    data: DataPtr<D>,
    unused_bytes: usize,
    // key is sttart address of used buffer
    // value is end address of used buffer
    used_buffer_range: BTreeMap<*mut u8, *mut u8>,
}

impl<D: DeviceBase, const N: usize> Default for StaticSizeBuffer<D, N> {
    fn default() -> Self {
        StaticSizeBuffer {
            data: DataPtr::new(N),
            unused_bytes: N,
            used_buffer_range: BTreeMap::new(),
        }
    }
}

impl<D: DeviceBase, const N: usize> StaticSizeBuffer<D, N> {
    fn last_used_buffer(&self) -> Option<(*mut u8, *mut u8)> {
        self.used_buffer_range
            .last()
            .map(|(start, end)| (*start, *end))
    }

    // 確保するメモリの始点と終点を返す
    fn start_end_ptr(&self) -> (*mut u8, *mut u8) {
        let (start, end) = if let Some((start, end)) = self.last_used_buffer() {
            // すでに確保済みの領域がある場合は、確保されている一番後ろの両浮に確保する
            // 確保する際は、MIDDLE_BUFFER_SIZE分だけ空白を明けてから確保する
            let start = unsafe { end.add(MIDDLE_BUFFER_SIZE) };
            let end = unsafe { start.add(bytes) };
            (start, end)
        } else {
            // ない場合は、先頭から確保する
            let start = self.data.ptr.as_ptr();
            let end = unsafe { start.add(bytes) };
            (start, end)
        };
    }

    fn alloc_update_unused_bytes(&mut self, bytes: usize) {
        // 確保した分とMIDDLE_BUFFER_SIZE分を未使用領域から減らす
        self.unused_bytes -= bytes + MIDDLE_BUFFER_SIZE;
    }

    fn free_update_unused_bytes(&mut self) {
        let (_, last_end) = self.last_used_buffer().unwrap();
        let num_byres_from_start_to_end = last_end as usize - self.data.ptr.as_ptr() as usize;
        self.unused_bytes = N - num_byres_from_start_to_end;
    }

    fn try_alloc(&mut self, bytes: usize) -> Result<*mut u8, ()> {
        if self.unused_bytes < bytes {
            return Err(());
        }

        let (start, end) = self.start_end_ptr();

        if end > unsafe { self.data.ptr.as_ptr().add(N) } {
            return Err(());
        }

        self.used_buffer_range.insert(start, end);
        self.alloc_update_unused_bytes(bytes);
    }

    fn try_free(&self, ptr: *mut u8) -> Result<(), ()> {
        self.used_buffer_range.remove(&ptr).ok_or(())?;
        self.free_update_unused_bytes();
        Ok(())
    }
}

#[cfg(test)]
mod static_buffer {
    use crate::{device::DeviceBase, num::Num};

    struct MockDeviceBase;

    impl DeviceBase for MockDeviceBase {
        fn zeros<T: Num>(len: usize) -> *mut T {
            todo!();
        }

        fn alloc(num_bytes: usize) -> *mut u8 {
            0 as *mut u8
        }

        fn drop_ptr<T>(ptr: *mut T, len: usize) {}

        fn get_item<T: Num>(ptr: *const T, offset: usize) -> T {
            todo!();
        }

        fn from_vec<T: Num>(vec: Vec<T>) -> *mut T {
            todo!();
        }

        fn clone_ptr<T>(ptr: *const T, len: usize) -> *mut T {
            todo!();
        }

        fn assign_item<T: Num>(ptr: *mut T, offset: usize, value: T) {
            todo!();
        }
    }
}

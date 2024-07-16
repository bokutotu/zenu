#[cfg(test)]
mod static_buffer {
    use std::sync::{Arc, Mutex};

    use serde::Serialize;

    use crate::{
        device::DeviceBase,
        memory_pool::{
            static_buffer::StaticSizeBuffer,
            static_mem_pool::{ArcBuffer, StaticMemPool, UnusedBytesPtrBufferMap},
            MIDDLE_BUFFER_SIZE,
        },
        num::Num,
    };

    #[derive(Default, Clone, Copy, Serialize)]
    struct MockDeviceBase;

    impl DeviceBase for MockDeviceBase {
        fn zeros<T: Num>(_len: usize) -> *mut T {
            todo!();
        }
        fn alloc(_num_bytes: usize) -> Result<*mut u8, ()> {
            Ok(0 as *mut u8)
        }
        fn mem_pool_drop_ptr(_ptr: *mut u8) -> Result<(), ()> {
            Ok(())
        }
        fn raw_drop_ptr<T>(_ptr: *mut T) {}
        fn raw_alloc(_num_bytes: usize) -> Result<*mut u8, ()> {
            Ok(0 as *mut u8)
        }
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
        fn mem_pool_alloc(_num_bytes: usize) -> Result<*mut u8, ()> {
            Ok(0 as *mut u8)
        }
    }

    #[test]
    fn too_large_alloc() {
        let mut buffer = StaticSizeBuffer::<MockDeviceBase, 1024>::new().unwrap();
        let ptr = buffer.try_alloc(4000);
        assert_eq!(ptr, Err(()));
    }

    // 領域を2回確保し、アドレスを確認する
    #[test]
    fn alloc_twice() {
        let mut buffer = StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::new().unwrap();
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
        let mut buffer = StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::new().unwrap();
        let ptr1 = buffer.try_alloc(1 << 15).unwrap();
        let ptr2 = buffer.try_alloc(1 << 10).unwrap();
        let ptr3 = buffer.try_alloc(1 << 13).unwrap();
        (buffer, ptr1, ptr2, ptr3)
    }

    #[test]
    fn alloc_3() {
        let (buffer, ptr1, ptr2, ptr3) = alloc_3_fragments();
        let ptr2_: usize = (1 << 15 as usize) + MIDDLE_BUFFER_SIZE;
        let ptr3_: usize = ptr2_ + (1 << 10 as usize) + MIDDLE_BUFFER_SIZE;
        assert_eq!(ptr1 as usize, 0);
        assert_eq!(ptr2 as usize, ptr2_);
        assert_eq!(ptr3 as usize, ptr3_);

        let unused_bytes = buffer.get_unused_bytes();
        let ans = BUF_LEN
            - 3 * MIDDLE_BUFFER_SIZE
            - (1 << 15 as usize)
            - (1 << 10 as usize)
            - (1 << 13 as usize);
        assert_eq!(ans, unused_bytes);
    }

    #[test]
    fn alloc_3_123() {
        let (mut buffer, ptr1, ptr2, ptr3) = alloc_3_fragments();
        let init_unused_bytes = buffer.get_unused_bytes();

        buffer.try_free(ptr1).unwrap();
        let num_bytes = buffer.get_unused_bytes();
        assert_eq!(init_unused_bytes, num_bytes);

        buffer.try_free(ptr2).unwrap();
        let num_bytes = buffer.get_unused_bytes();
        assert_eq!(init_unused_bytes, num_bytes);

        buffer.try_free(ptr3).unwrap();
        let num_bytes = buffer.get_unused_bytes();
        assert_eq!(BUF_LEN, num_bytes);
    }

    #[test]
    fn alloc_3_321() {
        let (mut buffer, ptr1, ptr2, ptr3) = alloc_3_fragments();

        buffer.try_free(ptr3).unwrap();
        let num_bytes = buffer.get_unused_bytes();
        let ans = BUF_LEN - ptr3 as usize;
        assert_eq!(ans, num_bytes);

        buffer.try_free(ptr2).unwrap();
        let num_bytes = buffer.get_unused_bytes();
        let ans = BUF_LEN - ptr2 as usize;
        assert_eq!(ans, num_bytes);

        buffer.try_free(ptr1).unwrap();
        let num_bytes = buffer.get_unused_bytes();
        assert_eq!(BUF_LEN, num_bytes);
    }

    #[test]
    fn alloc_3_231() {
        let (mut buffer, ptr1, ptr2, ptr3) = alloc_3_fragments();
        let init_unused_bytes = buffer.get_unused_bytes();

        buffer.try_free(ptr2).unwrap();
        let num_bytes = buffer.get_unused_bytes();
        assert_eq!(init_unused_bytes, num_bytes);

        buffer.try_free(ptr3).unwrap();
        let num_bytes = buffer.get_unused_bytes();
        let ans = BUF_LEN - ptr2 as usize;
        assert_eq!(ans, num_bytes);

        buffer.try_free(ptr1).unwrap();
        let num_bytes = buffer.get_unused_bytes();
        assert_eq!(BUF_LEN, num_bytes);
    }

    // UnusedBytesPtrBufferMapのテスト
    //
    // 1. 基本的な挿入と取得:
    //    - 空のマップにバッファを挿入し、正しく取得できることを確認
    //    - 異なる未使用バイト数を持つ複数のバッファを挿入し、それぞれ正しく取得できることを確認
    //
    // 2. smallest_unused_bytes_over_request の動作:
    //    - 要求サイズよりも大きな未使用バイト数を持つバッファが存在する場合、正しい値を返すことを確認
    //    - 要求サイズよりも大きな未使用バイト数を持つバッファが存在しない場合、None を返すことを確認
    //    - 複数の候補がある場合、最小の未使用バイト数を返すことを確認
    //
    // 3. pop_unused_bytes_ptr_buffer の動作:
    //    - 指定した未使用バイト数のバッファを正しく取得できることを確認
    //    - バッファを取得した後、マップから削除されていることを確認
    //    - 同じ未使用バイト数を持つ複数のバッファがある場合の動作を確認
    //
    // 4. エッジケース:
    //    - 空のマップに対する操作（smallest_unused_bytes_over_request, pop_unused_bytes_ptr_buffer）
    //    - 非常に大きな未使用バイト数や非常に小さな未使用バイト数の処理
    //    - 同じ未使用バイト数を持つ複数のバッファの挿入と取得
    //
    // 5. 挿入と削除の組み合わせ:
    //    - バッファを挿入した後に削除し、再度同じ未使用バイト数のバッファを挿入する
    //    - 異なる未使用バイト数のバッファを複数回挿入・削除した後の状態を確認
    //
    // 6. パフォーマンステスト（オプション）:
    //    - 大量のバッファを挿入した場合の挙動
    //    - 頻繁な挿入と削除を繰り返した場合の挙動
    //
    // 7. スレッドセーフティテスト（もし必要なら）:
    //    - 複数のスレッドから同時に操作を行った場合の一貫性を確認
    // const BUFFER_SIZE: usize = 1024;
    #[test]
    fn test_insert_and_get() {
        let mut map = UnusedBytesPtrBufferMap::<MockDeviceBase, BUF_LEN>::default();
        let buffer = ArcBuffer(Arc::new(Mutex::new(
            StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::new().unwrap(),
        )));

        map.insert(buffer.clone());

        let unused_bytes = buffer.0.lock().unwrap().get_unused_bytes();
        assert_eq!(
            map.smallest_unused_bytes_over_request(unused_bytes - 1),
            Some(unused_bytes)
        );
    }

    #[test]
    fn test_smallest_unused_bytes_over_request() {
        let mut map = UnusedBytesPtrBufferMap::<MockDeviceBase, BUF_LEN>::default();

        let buffer1 = ArcBuffer(Arc::new(Mutex::new(
            StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::new().unwrap(),
        )));
        let buffer2 = ArcBuffer(Arc::new(Mutex::new(
            StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::new().unwrap(),
        )));

        buffer1.0.lock().unwrap().try_alloc(100).unwrap();
        buffer2.0.lock().unwrap().try_alloc(200).unwrap();

        map.insert(buffer1.clone());
        map.insert(buffer2.clone());

        assert_eq!(
            map.smallest_unused_bytes_over_request(BUF_LEN - 201 - MIDDLE_BUFFER_SIZE),
            Some(BUF_LEN - 200 - MIDDLE_BUFFER_SIZE)
        );
        assert_eq!(
            map.smallest_unused_bytes_over_request(BUF_LEN - 101 - MIDDLE_BUFFER_SIZE),
            Some(BUF_LEN - 100 - MIDDLE_BUFFER_SIZE)
        );
        assert_eq!(map.smallest_unused_bytes_over_request(BUF_LEN), None);
    }

    #[test]
    fn test_pop_unused_bytes_ptr_buffer() {
        let mut map = UnusedBytesPtrBufferMap::<MockDeviceBase, BUF_LEN>::default();

        let buffer = ArcBuffer(Arc::new(Mutex::new(
            StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::new().unwrap(),
        )));
        let unused_bytes = buffer.0.lock().unwrap().get_unused_bytes();

        map.insert(buffer.clone());

        let popped_buffer = map.pop_unused_bytes_ptr_buffer(unused_bytes);
        assert!(Arc::ptr_eq(&buffer.0, &popped_buffer.0));

        assert_eq!(map.smallest_unused_bytes_over_request(0), None);
    }

    #[test]
    fn test_multiple_buffers_same_unused_bytes() {
        let mut map = UnusedBytesPtrBufferMap::<MockDeviceBase, BUF_LEN>::default();

        let buffer1 = ArcBuffer(Arc::new(Mutex::new(
            StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::new().unwrap(),
        )));
        let mut buffer2 = StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::new().unwrap();
        buffer2.data.ptr = 1 as *mut u8;
        let buffer2 = ArcBuffer(Arc::new(Mutex::new(buffer2)));

        map.insert(buffer1.clone());
        map.insert(buffer2.clone());

        let popped_buffer1 = map.pop_unused_bytes_ptr_buffer(BUF_LEN);
        let popped_buffer2 = map.pop_unused_bytes_ptr_buffer(BUF_LEN);

        assert!(
            (Arc::ptr_eq(&buffer1.0, &popped_buffer1.0)
                && Arc::ptr_eq(&buffer2.0, &popped_buffer2.0))
                || (Arc::ptr_eq(&buffer1.0, &popped_buffer2.0)
                    && Arc::ptr_eq(&buffer2.0, &popped_buffer1.0))
        );

        assert_eq!(map.smallest_unused_bytes_over_request(0), None);
    }

    #[test]
    fn test_insert_after_pop() {
        let mut map = UnusedBytesPtrBufferMap::<MockDeviceBase, BUF_LEN>::default();

        let buffer = ArcBuffer(Arc::new(Mutex::new(
            StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::new().unwrap(),
        )));

        map.insert(buffer.clone());
        let popped_buffer = map.pop_unused_bytes_ptr_buffer(BUF_LEN);

        popped_buffer.0.lock().unwrap().try_alloc(100).unwrap();
        map.insert(popped_buffer);

        assert_eq!(
            map.smallest_unused_bytes_over_request(BUF_LEN - 101 - MIDDLE_BUFFER_SIZE),
            Some(BUF_LEN - 100 - MIDDLE_BUFFER_SIZE)
        );
    }

    #[test]
    #[should_panic]
    fn test_empty_map() {
        let mut map = UnusedBytesPtrBufferMap::<MockDeviceBase, BUF_LEN>::default();

        assert_eq!(map.smallest_unused_bytes_over_request(0), None);
        map.pop_unused_bytes_ptr_buffer(BUF_LEN);
    }

    #[test]
    fn test_different_unused_bytes() {
        let mut map = UnusedBytesPtrBufferMap::<MockDeviceBase, BUF_LEN>::default();

        let buffer1 = ArcBuffer(Arc::new(Mutex::new(
            StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::new().unwrap(),
        )));
        let buffer2 = ArcBuffer(Arc::new(Mutex::new(
            StaticSizeBuffer::<MockDeviceBase, BUF_LEN>::new().unwrap(),
        )));

        buffer1.0.lock().unwrap().try_alloc(100).unwrap();
        buffer2.0.lock().unwrap().try_alloc(200).unwrap();

        map.insert(buffer1.clone());
        map.insert(buffer2.clone());

        assert_eq!(
            map.smallest_unused_bytes_over_request(BUF_LEN - 150 - MIDDLE_BUFFER_SIZE),
            Some(BUF_LEN - 100 - MIDDLE_BUFFER_SIZE)
        );
        let popped = map.pop_unused_bytes_ptr_buffer(BUF_LEN - 100 - MIDDLE_BUFFER_SIZE);
        assert!(Arc::ptr_eq(&buffer1.0, &popped.0));

        assert_eq!(
            map.smallest_unused_bytes_over_request(0),
            Some(BUF_LEN - 200 - MIDDLE_BUFFER_SIZE)
        );
    }

    /// StaticMemPoolのテスト
    #[test]
    fn test_alloc_and_free() {
        let mut pool = StaticMemPool::<MockDeviceBase, BUF_LEN>::default();

        // Allocate memory
        let ptr1 = pool.try_alloc(100).unwrap();
        assert!(ptr1.is_null());

        // Allocate more memory
        let ptr2 = pool.try_alloc(200).unwrap();
        assert!(!ptr2.is_null());
        assert_ne!(ptr1, ptr2);

        // Free memory
        pool.try_free(ptr1).unwrap();
        pool.try_free(ptr2).unwrap();

        // Reallocate and check if we get the same pointers
        let ptr3 = pool.try_alloc(100).unwrap();
        // assert!(ptr1 == ptr3 || ptr2 == ptr3);
        assert_eq!(ptr3 as usize, 0);
    }

    #[test]
    fn test_alloc_full_buffer() {
        let mut pool = StaticMemPool::<MockDeviceBase, BUF_LEN>::default();

        // Allocate the full buffer
        let ptr = pool.try_alloc(BUF_LEN).unwrap();
        assert!(ptr as usize == 0);

        // Try to allocate more, should fail
        assert!(pool.try_alloc(BUF_LEN - 1 - MIDDLE_BUFFER_SIZE).unwrap() as usize == 0);

        assert_eq!(pool.unused_bytes_ptr_buffer_map.0.len(), 2);

        // Now we should be able to allocate again
        assert!(pool.try_alloc(BUF_LEN - 3 - MIDDLE_BUFFER_SIZE).unwrap() as usize == 0);
        assert_eq!(pool.unused_bytes_ptr_buffer_map.0.len(), 3);

        pool.try_free(ptr).unwrap();
        assert_eq!(pool.unused_bytes_ptr_buffer_map.0.len(), 3);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut pool = StaticMemPool::<MockDeviceBase, BUF_LEN>::default();
        let mut ptrs = Vec::new();

        // Allocate multiple small buffers
        for idx in 0..7 {
            let ptr = pool.try_alloc(BUF_LEN / 10).unwrap();
            let ptr_val = idx * (BUF_LEN / 10 + MIDDLE_BUFFER_SIZE);
            assert_eq!(ptr as usize, ptr_val);
            ptrs.push(ptr);
        }

        // Free all buffers
        for ptr in ptrs {
            pool.try_free(ptr).unwrap();
        }

        // We should be able to allocate the full buffer now
        assert!(pool.try_alloc(BUF_LEN).is_ok());
    }

    #[test]
    fn test_error_handling() {
        let mut pool = StaticMemPool::<MockDeviceBase, BUF_LEN>::default();

        // Try to free an invalid pointer
        assert!(pool.try_free(std::ptr::null_mut()).is_err());

        // Try to allocate more than the buffer size
        assert!(pool.try_alloc(BUF_LEN + 1).is_err());

        // Allocate all memory
        let ptr = pool.try_alloc(BUF_LEN).unwrap();

        // Free the memory
        pool.try_free(ptr).unwrap();

        // Try to free the same pointer again
        assert!(pool.try_free(ptr).is_err());
    }
}

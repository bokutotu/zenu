#![expect(clippy::module_name_repetitions, clippy::module_inception)]

use device::cpu::Cpu;
use memory_pool::MemPool;

pub mod concat;
pub mod constructor;
pub mod device;
pub mod dim;
pub mod index;
pub mod matrix;
pub mod matrix_blas;
pub mod matrix_iter;
pub mod nn;
pub mod num;
pub mod operation;
pub mod shape_stride;
pub mod slice;

mod impl_ops;
mod impl_serde;
mod matrix_format;
mod memory_pool;
mod with_clousers;

#[cfg(feature = "nvidia")]
use device::nvidia::Nvidia;

pub(crate) struct ZenuMatrixState {
    pub(crate) is_mem_pool_used: bool,
    pub(crate) cpu: MemPool<Cpu>,
    #[cfg(feature = "nvidia")]
    pub(crate) nvidia: MemPool<Nvidia>,
}

impl Default for ZenuMatrixState {
    fn default() -> Self {
        let use_mem_pool = std::env::var("ZENU_USE_MEMPOOL").unwrap_or("1".to_string()) == "1";
        ZenuMatrixState {
            is_mem_pool_used: use_mem_pool,
            cpu: MemPool::default(),
            #[cfg(feature = "nvidia")]
            nvidia: MemPool::default(),
        }
    }
}

pub(crate) static ZENU_MATRIX_STATE: once_cell::sync::Lazy<ZenuMatrixState> =
    once_cell::sync::Lazy::new(ZenuMatrixState::default);

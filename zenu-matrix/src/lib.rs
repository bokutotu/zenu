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

#[cfg(feature = "nvidia")]
use device::nvidia::Nvidia;

pub(crate) struct ZenuMatrixState {
    pub(crate) use_mem_pool: bool,
    pub(crate) cpu_mem_pool: MemPool<Cpu>,
    #[cfg(feature = "nvidia")]
    pub(crate) nvidia_mem_pool: MemPool<Nvidia>,
}

impl Default for ZenuMatrixState {
    fn default() -> Self {
        let use_mem_pool = std::env::var("ZENU_USE_MEMPOOL").unwrap_or("1".to_string()) == "1";
        ZenuMatrixState {
            use_mem_pool,
            cpu_mem_pool: MemPool::default(),
            #[cfg(feature = "nvidia")]
            nvidia_mem_pool: MemPool::default(),
        }
    }
}

pub(crate) static ZENU_MATRIX_STATE: once_cell::sync::Lazy<ZenuMatrixState> =
    once_cell::sync::Lazy::new(|| ZenuMatrixState::default());

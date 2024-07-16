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
    pub(crate) cpu_mem_pool: MemPool<Cpu>,
    #[cfg(feature = "nvidia")]
    pub(crate) nvidia_mem_pool: MemPool<Nvidia>,
}

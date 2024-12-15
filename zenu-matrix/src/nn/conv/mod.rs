pub mod bkwd_data;
pub mod bkwd_filter;
pub mod fwd;
pub mod interface;

#[cfg(feature = "nvidia")]
mod nvidia;

mod cpu;
mod utils;

mod cudnn;
mod naive;

pub use cudnn::lstm_cudnn;
pub use naive::lstm_naive;

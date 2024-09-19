mod descriptor;
mod helper;
mod test;

pub use helper::{RNNAlgo, RNNBias, RNNCell, RNNDataLayout, RNNMathType};

pub use descriptor::{GRUParams, LSTMParams, RNNContext, RNNDescriptor, RNNParams};

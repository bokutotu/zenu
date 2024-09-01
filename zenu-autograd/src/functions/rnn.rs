use zenu_matrix::{nn::rnn::RNNDescriptor, num::Num};

struct CudnnRNN<T: Num> {
    rnn_desc: RNNDescriptor<T>,
}

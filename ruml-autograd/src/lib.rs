pub trait Layer {
    type Forward;
    type Backward;

    fn forward_node(&self) -> Self::Forward;
    fn backward_node(&self) -> Self::Backward;
}

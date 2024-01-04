use ruml_graph::{Graph, Node};

pub trait Layer {
    type Forward: Node;
    type Backward: Node;

    fn forward_node(&self) -> Self::Forward;
    fn backward_node(&self) -> Self::Backward;
}

pub trait AutoGradGraph: Graph + Node {
    fn forward(&self);
    fn backward(&self);
}

pub struct Sequential {
    layers: Vec<Box<dyn Node>>,
}

pub trait Graph {
    type Input;
    type Output;

    fn init(&self);
    fn cal(&self);
    fn set_input(&self, input: Self::Input);
    fn get_output(&self) -> Self::Output;
    fn add_node(&self, node: Box<dyn Node<Input = Self::Input, Output = Self::Output>>);
}

pub trait Node {
    type Input;
    type Output;

    fn set_input(&self, input: Self::Input);
    fn get_output(&self) -> Self::Output;
    fn cal(&self);
}

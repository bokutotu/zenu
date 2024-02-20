use ruml_matrix::{
    dim::DimTrait,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    memory::OwnedMemory,
    operation::add::MatrixAdd,
};

use crate::{Function, Variable, VariableWeak};

pub struct Add<M: OwnedMemory> {
    x: Variable<M>,
    y: Variable<M>,
    output: VariableWeak<M>,
}

impl<M: OwnedMemory> Add<M> {
    pub fn new(x: Variable<M>, y: Variable<M>, output: Variable<M>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<M: OwnedMemory> Function<M> for Add<M> {
    fn forward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        if x.shape().len() > y.shape().len() && x.shape() == output.shape() {
            MatrixAdd::add(output.to_view_mut(), x.to_view(), y.to_view());
        } else if y.shape().len() > x.shape().len() && y.shape() == output.shape() {
            MatrixAdd::add(output.to_view_mut(), y.to_view(), x.to_view());
        } else {
            panic!("Invalid shapes");
        }
    }

    fn backward(&self) {
        todo!();
    }

    fn get_inputs(&self) -> Vec<Variable<M>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

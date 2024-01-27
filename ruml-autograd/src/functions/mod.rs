pub mod add;

use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::variable::Variable;
use crate::Value;

pub trait Function<V: Value> {
    fn forward(&self);
    fn backward(&self);
    fn get_inputs(&self) -> Vec<Variable<V>>;
    fn get_gen(&self) -> usize {
        let inputs = self.get_inputs();
        inputs
            .iter()
            .map(|input| input.get_gen())
            .max()
            .unwrap_or(0)
    }
}

#[derive(Clone)]
pub(crate) struct FunctionQueueItem<V> {
    pub(crate) func: Rc<RefCell<Box<dyn Function<V>>>>,
    pub(crate) gen: usize,
}

impl<V: Value> From<Rc<RefCell<Box<dyn Function<V>>>>> for FunctionQueueItem<V> {
    fn from(func: Rc<RefCell<Box<dyn Function<V>>>>) -> Self {
        Self {
            func: func.clone(),
            gen: func.borrow().get_gen(),
        }
    }
}

impl<V> PartialEq for FunctionQueueItem<V> {
    fn eq(&self, other: &Self) -> bool {
        self.gen == other.gen
    }
}

impl<V> Eq for FunctionQueueItem<V> {
    fn assert_receiver_is_total_eq(&self) {}
}

impl<V> PartialOrd for FunctionQueueItem<V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.gen.cmp(&other.gen))
    }
}

impl<V> Ord for FunctionQueueItem<V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gen.cmp(&other.gen)
    }
}

impl<V> Deref for FunctionQueueItem<V> {
    type Target = Rc<RefCell<Box<dyn Function<V>>>>;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

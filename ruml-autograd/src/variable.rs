use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashSet},
    rc::{Rc, Weak},
};

use crate::{
    functions::{add::add, Function, FunctionQueueItem},
    value::Value,
};

pub struct VariableInner<V> {
    data: V,
    creator: Option<Rc<RefCell<Box<dyn Function<V>>>>>,
    grad: Option<Variable<V>>,
    gen: usize,
    name: Option<String>,
}

impl<V: Value> VariableInner<V> {
    pub fn new(data: V) -> Self {
        Self {
            data,
            creator: None,
            grad: None,
            gen: 0,
            name: None,
        }
    }

    pub fn get_data(&self) -> V {
        self.data.clone()
    }

    pub fn set_data(&mut self, data: V) {
        self.data = data;
    }

    pub fn set_creator(&mut self, creator: Rc<RefCell<Box<dyn Function<V>>>>) {
        self.creator = Some(creator);
        let gen = self.creator.as_ref().unwrap().borrow().get_gen();
        self.gen = gen + 1;
    }

    pub fn get_creator(&self) -> &Option<Rc<RefCell<Box<dyn Function<V>>>>> {
        &self.creator
    }

    pub fn get_grad(&self) -> &Option<Variable<V>> {
        &self.grad
    }

    pub fn set_grad(&mut self, grad: Variable<V>) {
        match self.grad {
            Some(ref grad_) => {
                let new_grad = add(grad_, &grad);
                self.grad = Some(new_grad);
            }
            None => {
                self.grad = Some(grad);
            }
        }
    }

    pub fn backward(&self) {
        let mut funcs: BinaryHeap<FunctionQueueItem<V>> = BinaryHeap::new();
        let mut seen_rc = HashSet::new();

        funcs.push(self.creator.clone().unwrap().into());

        while let Some(FunctionQueueItem { func, .. }) = funcs.pop() {
            func.borrow().backward();
            func.borrow().get_inputs().iter().for_each(|input| {
                if let Some(creator) = input.get_creator() {
                    if !seen_rc.contains(&creator.as_ptr()) {
                        funcs.push(creator.clone().into());
                        seen_rc.insert(creator.as_ptr());
                    }
                }
            });
        }
    }

    pub fn get_gen(&self) -> usize {
        self.gen
    }

    pub fn set_gen(&mut self, gen: usize) {
        self.gen = gen;
    }

    pub fn clear_grad(&mut self) {
        if let Some(ref mut grad) = self.grad {
            *grad = Variable::new(V::zero());
        }
    }

    pub fn set_name(&mut self, name: &str) {
        self.name = Some(name.to_string());
    }

    pub fn get_name(&self) -> &Option<String> {
        &self.name
    }
}

#[derive(Clone)]
pub struct Variable<V> {
    inner: Rc<RefCell<VariableInner<V>>>,
}

impl<V: Value> Variable<V> {
    pub fn new(data: V) -> Self {
        let inner = VariableInner::new(data);
        let inner = Rc::new(RefCell::new(inner));
        Self { inner }
    }

    pub fn get_data(&self) -> V {
        self.inner.borrow().get_data()
    }

    pub fn set_data(&self, data: V) {
        self.inner.borrow_mut().set_data(data);
    }

    pub fn set_creator(&self, creator: Rc<RefCell<Box<dyn Function<V>>>>) {
        self.inner.borrow_mut().set_creator(creator);
    }

    pub fn get_creator(&self) -> Option<Rc<RefCell<Box<dyn Function<V>>>>> {
        self.inner.borrow().get_creator().clone()
    }

    pub fn get_grad(&self) -> Option<Variable<V>> {
        self.inner.borrow().get_grad().clone()
    }

    pub fn set_grad(&self, grad: Variable<V>) {
        self.inner.borrow_mut().set_grad(grad);
    }

    pub fn backward(&self) {
        if self.inner.borrow().get_grad().is_none() {
            self.inner.borrow_mut().set_grad(Variable::new(V::one()));
        }
        self.inner.borrow().backward();
    }

    pub fn downgrade(self) -> VariableWeak<V> {
        VariableWeak {
            inner: Rc::downgrade(&self.inner),
        }
    }

    pub fn get_gen(&self) -> usize {
        self.inner.borrow().get_gen()
    }

    pub fn clear_grad(&self) {
        self.inner.borrow_mut().clear_grad();
    }

    pub fn set_name(&self, name: &str) {
        self.inner.borrow_mut().set_name(name);
    }

    pub fn get_name(&self) -> Option<String> {
        self.inner.borrow().get_name().clone()
    }
}

impl<V> AsRef<Variable<V>> for Variable<V> {
    fn as_ref(&self) -> &Variable<V> {
        self
    }
}

#[derive(Debug, Clone)]
pub struct VariableWeak<V> {
    inner: Weak<RefCell<VariableInner<V>>>,
}

impl<V> VariableWeak<V> {
    pub fn upgrade(&self) -> Option<Variable<V>> {
        self.inner.upgrade().map(|inner| Variable { inner })
    }
}

use std::{
    cell::{Ref, RefCell, RefMut},
    collections::{BinaryHeap, HashSet},
    ops::Deref,
    rc::{Rc, Weak},
};

use ruml_matrix::{
    matrix::MatrixBase,
    matrix_impl::CpuOwnedMatrixDyn,
    num::Num,
    operation::{ones::Ones, zeros::Zeros},
};

pub trait Function<T: Num> {
    fn forward(&self);
    fn backward(&self);
    fn get_inputs(&self) -> Vec<Variable<T>>;
    fn get_gen(&self) -> usize {
        let inputs = self.get_inputs();
        inputs.iter().map(|input| input.get_gen()).max().unwrap()
    }
}

#[derive(Clone)]
pub(crate) struct FunctionQueueItem<V> {
    pub(crate) func: Rc<RefCell<Box<dyn Function<V>>>>,
    pub(crate) gen: usize,
}

impl<V: Num> From<Rc<RefCell<Box<dyn Function<V>>>>> for FunctionQueueItem<V> {
    fn from(func: Rc<RefCell<Box<dyn Function<V>>>>) -> Self {
        Self {
            func: func.clone(),
            gen: func.borrow().get_gen(),
        }
    }
}

impl<V: Num> PartialEq for FunctionQueueItem<V> {
    fn eq(&self, other: &Self) -> bool {
        self.gen == other.gen
    }
}

impl<V: Num> Eq for FunctionQueueItem<V> {
    fn assert_receiver_is_total_eq(&self) {}
}

impl<V: Num> PartialOrd for FunctionQueueItem<V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.gen.cmp(&other.gen))
    }
}

impl<V: Num> Ord for FunctionQueueItem<V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gen.cmp(&other.gen)
    }
}

impl<T: Num> Deref for FunctionQueueItem<T> {
    type Target = Rc<RefCell<Box<dyn Function<T>>>>;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

#[derive(Clone)]
pub enum Tensor<T: Num> {
    Cpu(CpuOwnedMatrixDyn<T>),
    Gpu(),
}

impl<T: Num> Tensor<T> {
    pub fn zeros_like(&self) -> Self {
        match self {
            Tensor::Cpu(tensor) => {
                let shape = tensor.shape();
                Tensor::Cpu(CpuOwnedMatrixDyn::zeros(shape))
            }
            Tensor::Gpu() => Tensor::Gpu(),
        }
    }

    pub fn ones_like(&self) -> Self {
        match self {
            Tensor::Cpu(tensor) => {
                let shape = tensor.shape();
                Tensor::Cpu(CpuOwnedMatrixDyn::ones(shape))
            }
            Tensor::Gpu() => Tensor::Gpu(),
        }
    }
}

#[derive(Clone)]
pub struct VariableInner<T: Num> {
    data: Tensor<T>,
    creator: Option<Rc<RefCell<Box<dyn Function<T>>>>>,
    grad: Option<Variable<T>>,
    gen: usize,
    name: Option<String>,
}

impl<T: Num> VariableInner<T> {
    pub fn new(data: Tensor<T>) -> Self {
        VariableInner {
            data,
            creator: None,
            grad: None,
            gen: 0,
            name: None,
        }
    }

    #[allow(clippy::type_complexity)]
    fn get_creator(&self) -> Option<Rc<RefCell<Box<dyn Function<T>>>>> {
        self.creator.clone()
    }

    fn set_creator(&mut self, creator: Rc<RefCell<Box<dyn Function<T>>>>) {
        self.creator = Some(creator);
    }

    fn get_grad(&self) -> Option<Variable<T>> {
        self.grad.clone()
    }

    fn set_grad(&mut self, grad: Variable<T>) {
        self.grad = Some(grad);
    }

    fn get_gen(&self) -> usize {
        self.gen
    }

    fn set_gen(&mut self, gen: usize) {
        self.gen = gen;
    }

    fn get_name(&self) -> Option<String> {
        self.name.clone()
    }

    fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    pub fn backward(&self) {
        let mut funcs: BinaryHeap<FunctionQueueItem<T>> = BinaryHeap::new();
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

    fn clear_grad(&mut self) {
        if let Some(ref mut grad) = self.grad {
            grad.inner.borrow_mut().clear_grad();
        }
    }
}

#[derive(Clone)]
pub struct Variable<T: Num> {
    inner: Rc<RefCell<VariableInner<T>>>,
}

impl<T: Num> Variable<T> {
    pub fn new(data: Tensor<T>) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VariableInner::new(data))),
        }
    }
    pub fn get_data<'a>(&'a self) -> Ref<'a, Tensor<T>> {
        let reference: Ref<'a, VariableInner<T>> = self.inner.borrow();
        Ref::map(reference, |r| &r.data)
    }

    pub fn get_data_mut<'a>(&'a self) -> RefMut<'a, Tensor<T>> {
        let reference: RefMut<'a, VariableInner<T>> = self.inner.borrow_mut();
        RefMut::map(reference, |r| &mut r.data)
    }

    pub fn set_creator(&self, creator: Rc<RefCell<Box<dyn Function<T>>>>) {
        self.inner.borrow_mut().set_creator(creator);
    }

    pub fn get_creator(&self) -> Option<Rc<RefCell<Box<dyn Function<T>>>>> {
        self.inner.borrow().get_creator().clone()
    }

    pub fn get_grad<'a>(&'a self) -> Ref<'a, Option<Variable<T>>> {
        let reference: Ref<'a, VariableInner<T>> = self.inner.borrow();
        Ref::map(reference, |r| &r.grad)
    }

    pub fn get_grad_mut<'a>(&'a self) -> RefMut<'a, Option<Variable<T>>> {
        let reference: RefMut<'a, VariableInner<T>> = self.inner.borrow_mut();
        RefMut::map(reference, |r| &mut r.grad)
    }

    pub fn backward(&self) {
        if self.inner.borrow().get_grad().is_none() {
            let zeros = self.get_data().zeros_like();
            let zeros = Variable::new(zeros);
            self.inner.borrow_mut().set_grad(zeros);
        }
        self.inner.borrow().backward();
    }

    pub fn downgrade(self) -> VariableWeak<T> {
        VariableWeak {
            inner: Rc::downgrade(&self.inner),
        }
    }

    pub fn get_gen(&self) -> usize {
        self.inner.borrow().get_gen()
    }

    pub fn set_gen(&self, gen: usize) {
        self.inner.borrow_mut().set_gen(gen);
    }

    pub fn clear_grad(&self) {
        self.inner.borrow_mut().clear_grad();
    }

    pub fn set_name(&self, name: &str) {
        self.inner.borrow_mut().set_name(name.to_string());
    }

    pub fn get_name(&self) -> Option<String> {
        self.inner.borrow().get_name().clone()
    }
}

#[derive(Debug, Clone)]
pub struct VariableWeak<T: Num> {
    inner: Weak<RefCell<VariableInner<T>>>,
}

impl<T: Num> VariableWeak<T> {
    pub fn upgrade(&self) -> Option<Variable<T>> {
        self.inner.upgrade().map(|inner| Variable { inner })
    }
}

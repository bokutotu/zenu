pub mod functions;

use std::{
    cell::{Ref, RefCell, RefMut},
    collections::{BinaryHeap, HashSet},
    ops::Deref,
    rc::{Rc, Weak},
};

use zenu_matrix::{
    dim::DimDyn,
    matrix::{MatrixBase, OwnedMatrix},
    matrix_impl::Matrix,
    memory_impl::OwnedMem,
    num::Num,
    operation::ones::Ones,
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
pub(crate) struct FunctionQueueItem<T: Num> {
    pub(crate) func: Rc<RefCell<Box<dyn Function<T>>>>,
    pub(crate) gen: usize,
}

impl<T: Num> From<Rc<RefCell<Box<dyn Function<T>>>>> for FunctionQueueItem<T> {
    fn from(func: Rc<RefCell<Box<dyn Function<T>>>>) -> Self {
        Self {
            func: func.clone(),
            gen: func.borrow().get_gen(),
        }
    }
}

impl<T: Num> PartialEq for FunctionQueueItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.gen == other.gen
    }
}

impl<T: Num> Eq for FunctionQueueItem<T> {
    fn assert_receiver_is_total_eq(&self) {}
}

impl<T: Num> PartialOrd for FunctionQueueItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.gen.cmp(&other.gen))
    }
}

impl<T: Num> Ord for FunctionQueueItem<T> {
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
pub struct VariableInner<T: Num> {
    data: Matrix<OwnedMem<T>, DimDyn>,
    creator: Option<Rc<RefCell<Box<dyn Function<T>>>>>,
    grad: Option<Variable<T>>,
    gen: usize,
    name: Option<String>,
}

impl<T: Num> VariableInner<T> {
    pub fn new(data: Matrix<OwnedMem<T>, DimDyn>) -> Self {
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
        let gen = self.creator.as_ref().unwrap().borrow().get_gen();
        self.gen = gen + 1;
    }

    fn get_gen(&self) -> usize {
        self.gen
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

impl<T: Num> From<T> for Variable<T> {
    fn from(data: T) -> Self {
        let data = Matrix::from_vec(vec![data], DimDyn::new(&[]));
        Variable::new(data)
    }
}

impl<T: Num> From<Matrix<OwnedMem<T>, DimDyn>> for Variable<T> {
    fn from(data: Matrix<OwnedMem<T>, DimDyn>) -> Self {
        Variable::new(data)
    }
}

impl<T: Num> Variable<T> {
    pub fn new(data: Matrix<OwnedMem<T>, DimDyn>) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VariableInner::new(data))),
        }
    }
    pub fn get_data<'a>(&'a self) -> Ref<'a, Matrix<OwnedMem<T>, DimDyn>> {
        let reference: Ref<'a, VariableInner<T>> = self.inner.borrow();
        Ref::map(reference, |r| &r.data)
    }

    pub fn get_data_mut<'a>(&'a self) -> RefMut<'a, Matrix<OwnedMem<T>, DimDyn>> {
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
        if self.inner.borrow().grad.is_none() {
            let ones = Ones::ones(self.get_data().shape());
            let ones = Variable::new(ones);
            self.inner.borrow_mut().grad = Some(ones);
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

    pub fn clear_grad(&self) {
        self.inner.borrow_mut().clear_grad();
    }

    pub fn set_name(&self, name: &str) {
        self.inner.borrow_mut().set_name(name.to_string());
    }

    pub fn get_name(&self) -> Option<String> {
        self.inner.borrow().get_name().clone()
    }

    pub fn with_grad_data<F>(&self, mut f: F)
    where
        F: FnMut(&Matrix<OwnedMem<T>, DimDyn>),
    {
        let inner = self.inner.borrow();
        if let Some(grad_variable) = &inner.grad {
            let grad_inner = grad_variable.inner.borrow();
            f(&grad_inner.data);
        } else {
            panic!("grad is None");
        }
    }

    pub fn set_grad(&self, grad: Variable<T>) {
        *self.get_grad_mut() = Some(grad);
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
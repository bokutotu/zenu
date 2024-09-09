pub mod concat;
pub mod creator;
pub mod functions;

use std::{
    cell::{Ref, RefCell, RefMut},
    collections::{BinaryHeap, HashSet},
    fmt::{Debug, Display},
    ops::Deref,
    rc::{Rc, Weak},
    sync::Mutex,
};

use creator::{ones::ones, zeros::zeros_like};
use functions::sum_to::sum_to;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use zenu_matrix::{
    device::Device,
    dim::{larger_shape, DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref as MRef},
    num::Num,
};

pub trait Function<T: Num, D: Device> {
    fn forward(&self);
    fn backward(&self);
    fn get_inputs(&self) -> Vec<Variable<T, D>>;
    fn get_gen(&self) -> usize {
        let inputs = self.get_inputs();
        inputs.iter().map(|input| input.get_gen()).max().unwrap()
    }
}

lazy_static! {
    static ref IS_TRAIN: Mutex<bool> = Mutex::new(true);
}

pub fn no_train() {
    let mut is_train = IS_TRAIN.lock().unwrap();
    *is_train = false;
}

pub fn is_train() -> bool {
    let is_train = IS_TRAIN.lock().unwrap();
    *is_train
}

pub fn set_train() {
    let mut is_train = IS_TRAIN.lock().unwrap();
    *is_train = true;
}

#[derive(Clone)]
pub(crate) struct FunctionQueueItem<T: Num, D: Device> {
    pub(crate) func: Rc<RefCell<Box<dyn Function<T, D>>>>,
    pub(crate) gen: usize,
}

impl<T: Num, D: Device> From<Rc<RefCell<Box<dyn Function<T, D>>>>> for FunctionQueueItem<T, D> {
    fn from(func: Rc<RefCell<Box<dyn Function<T, D>>>>) -> Self {
        Self {
            func: func.clone(),
            gen: func.borrow().get_gen(),
        }
    }
}

impl<T: Num, D: Device> PartialEq for FunctionQueueItem<T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.gen == other.gen
    }
}

impl<T: Num, D: Device> Eq for FunctionQueueItem<T, D> {
    fn assert_receiver_is_total_eq(&self) {}
}

impl<T: Num, D: Device> PartialOrd for FunctionQueueItem<T, D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.gen.cmp(&other.gen))
    }
}

impl<T: Num, D: Device> Ord for FunctionQueueItem<T, D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gen.cmp(&other.gen)
    }
}

impl<T: Num, D: Device> Deref for FunctionQueueItem<T, D> {
    type Target = Rc<RefCell<Box<dyn Function<T, D>>>>;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

#[derive(Clone)]
pub struct VariableInner<T: Num, D: Device> {
    data: Matrix<Owned<T>, DimDyn, D>,
    #[expect(clippy::type_complexity)]
    creator: Option<Rc<RefCell<Box<dyn Function<T, D>>>>>,
    grad: Option<Variable<T, D>>,
    gen: usize,
    name: Option<String>,
    is_train: bool,
}

impl<T, D> Serialize for VariableInner<T, D>
where
    T: Num,
    D: Device,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

impl<'de, T, D> Deserialize<'de> for VariableInner<T, D>
where
    T: Num + Deserialize<'de>,
    D: Device,
{
    fn deserialize<Ds>(deserializer: Ds) -> Result<Self, Ds::Error>
    where
        Ds: serde::Deserializer<'de>,
    {
        let data = Matrix::<Owned<T>, DimDyn, D>::deserialize(deserializer)?;
        Ok(VariableInner {
            data,
            creator: None,
            grad: None,
            gen: 0,
            name: None,
            is_train: false,
        })
    }
}

impl<T: Num, D: Device> VariableInner<T, D> {
    pub fn new(data: Matrix<Owned<T>, DimDyn, D>) -> Self {
        VariableInner {
            data,
            creator: None,
            grad: None,
            gen: 0,
            name: None,
            is_train: false,
        }
    }

    #[expect(clippy::type_complexity)]
    fn get_creator(&self) -> Option<Rc<RefCell<Box<dyn Function<T, D>>>>> {
        self.creator.clone()
    }

    fn set_creator(&mut self, creator: Rc<RefCell<Box<dyn Function<T, D>>>>) {
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
        let mut funcs: BinaryHeap<FunctionQueueItem<T, D>> = BinaryHeap::new();
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
        self.grad = None;
    }

    fn get_is_train(&self) -> bool {
        self.is_train
    }

    fn set_is_train(&mut self, is_train: bool) {
        self.is_train = is_train;
    }

    fn get_all_variable(&self) -> Vec<Variable<T, D>> {
        let mut variables = Vec::new();
        let mut seen_rc = HashSet::new();
        let mut funcs: BinaryHeap<FunctionQueueItem<T, D>> = BinaryHeap::new();

        funcs.push(self.creator.clone().unwrap().into());

        while let Some(FunctionQueueItem { func, .. }) = funcs.pop() {
            let inputs = func.borrow().get_inputs();
            for input in inputs {
                if let Some(creator) = input.get_creator() {
                    if !seen_rc.contains(&creator.as_ptr()) {
                        funcs.push(creator.clone().into());
                        seen_rc.insert(creator.as_ptr());
                    }
                }
            }
            let inputs = func.borrow().get_inputs();
            for input in inputs {
                variables.push(input);
            }
        }

        variables.dedup_by(|a, b| a.get_data().as_ptr() == b.get_data().as_ptr());
        variables
    }

    fn get_all_trainable_variables(&self) -> Vec<Variable<T, D>> {
        let variables = self.get_all_variable();
        variables.into_iter().filter(|v| v.get_is_train()).collect()
    }

    fn to<DO: Device>(&self) -> VariableInner<T, DO> {
        if self.grad.is_some() {
            panic!("grad must be None");
        }
        VariableInner {
            data: self.data.new_matrix().to(),
            creator: None,
            grad: None,
            gen: 0,
            name: self.name.clone(),
            is_train: self.is_train,
        }
    }
}

#[derive(Clone)]
pub struct Variable<T: Num, D: Device> {
    inner: Rc<RefCell<VariableInner<T, D>>>,
}

impl<T: Num, D: Device> From<T> for Variable<T, D> {
    fn from(data: T) -> Self {
        let data = Matrix::from_vec(vec![data], DimDyn::new(&[]));
        Variable::new(data)
    }
}

impl<T: Num, D: Device> From<Matrix<Owned<T>, DimDyn, D>> for Variable<T, D> {
    fn from(data: Matrix<Owned<T>, DimDyn, D>) -> Self {
        Variable::new(data)
    }
}

impl<T, D> Serialize for Variable<T, D>
where
    T: Num,
    D: Device,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.inner.borrow().clone().serialize(serializer)
    }
}

impl<'de, T, D> Deserialize<'de> for Variable<T, D>
where
    T: Num + Deserialize<'de>,
    D: Device,
{
    fn deserialize<Ds>(deserializer: Ds) -> Result<Self, Ds::Error>
    where
        Ds: serde::Deserializer<'de>,
    {
        let inner = VariableInner::<T, D>::deserialize(deserializer)?;
        Ok(Variable {
            inner: Rc::new(RefCell::new(inner)),
        })
    }
}

impl<T: Num, D: Device> Variable<T, D> {
    pub fn new(data: Matrix<Owned<T>, DimDyn, D>) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VariableInner::new(data))),
        }
    }

    pub fn swap_inner(&self, inner: Matrix<Owned<T>, DimDyn, D>) {
        self.inner.borrow_mut().data = inner;
    }

    pub fn get_data<'a>(&'a self) -> Ref<'a, Matrix<Owned<T>, DimDyn, D>> {
        let reference: Ref<'a, VariableInner<T, D>> = self.inner.borrow();
        Ref::map(reference, |r| &r.data)
    }

    pub fn get_as_ref<'a>(&self) -> Matrix<MRef<&'a T>, DimDyn, D> {
        let data = self.get_data();
        data.to_ref()
    }

    pub fn get_as_mut<'a>(&self) -> Matrix<MRef<&'a mut T>, DimDyn, D> {
        let mut data = self.get_data_mut();
        data.to_ref_mut()
    }

    pub fn get_data_mut<'a>(&'a self) -> RefMut<'a, Matrix<Owned<T>, DimDyn, D>> {
        let reference: RefMut<'a, VariableInner<T, D>> = self.inner.borrow_mut();
        RefMut::map(reference, |r| &mut r.data)
    }

    pub fn set_creator(&self, creator: Rc<RefCell<Box<dyn Function<T, D>>>>) {
        self.inner.borrow_mut().set_creator(creator);
    }

    #[expect(clippy::type_complexity)]
    pub fn get_creator(&self) -> Option<Rc<RefCell<Box<dyn Function<T, D>>>>> {
        self.inner.borrow().get_creator().clone()
    }

    pub fn get_grad<'a>(&'a self) -> Option<Variable<T, D>> {
        let reference: Ref<'a, VariableInner<T, D>> = self.inner.borrow();
        let ref_option = Ref::map(reference, |r| &r.grad);
        ref_option.clone()
    }

    fn get_grad_mut<'a>(&'a self) -> RefMut<'a, Option<Variable<T, D>>> {
        let reference: RefMut<'a, VariableInner<T, D>> = self.inner.borrow_mut();
        RefMut::map(reference, |r| &mut r.grad)
    }

    pub fn backward(&self) {
        if self.inner.borrow().grad.is_none() {
            let ones = ones(self.get_data().shape());
            ones.set_name(&format!("{:?}_grad", self.get_name().unwrap_or_default()));
            self.inner.borrow_mut().grad = Some(ones);
        }
        self.inner.borrow().backward();
    }

    pub fn downgrade(self) -> VariableWeak<T, D> {
        VariableWeak {
            inner: Rc::downgrade(&self.inner),
        }
    }

    pub fn get_gen(&self) -> usize {
        self.inner.borrow().get_gen()
    }

    pub fn clear_grad(&self) {
        self.inner.borrow_mut().clear_grad();
        let all_val = self.inner.borrow().get_all_variable();
        for val in all_val {
            val.inner.borrow_mut().clear_grad();
        }
    }

    pub fn set_name(&self, name: &str) {
        self.inner.borrow_mut().set_name(name.to_string());
    }

    pub fn get_name(&self) -> Option<String> {
        self.inner.borrow().get_name().clone()
    }

    pub fn with_grad_data<F>(&self, mut f: F)
    where
        F: FnMut(&Matrix<Owned<T>, DimDyn, D>),
    {
        let inner = self.inner.borrow();
        if let Some(grad_variable) = &inner.grad {
            let grad_inner = grad_variable.inner.borrow();
            f(&grad_inner.data);
        } else {
            panic!("grad is None");
        }
    }

    pub fn set_grad(&self, mut grad: Variable<T, D>) {
        let self_shape = self.get_shape();
        let grad_shape = grad.get_shape();
        let larger_shape_ = larger_shape(self_shape, grad_shape);
        if self_shape.slice() == grad_shape.slice() {
        } else if self_shape.slice() == larger_shape_.slice() {
            grad = zeros_like(self) + grad;
        } else if grad_shape.slice() == larger_shape_.slice() {
            grad = sum_to(grad, self_shape);
        } else {
            panic!("shape of grad and data must be same");
        }
        let name = self.get_name().clone().unwrap_or_default();
        let mut grad_mut = self.get_grad_mut();
        match *grad_mut {
            Some(ref mut grad_variable) => {
                *grad_variable = grad + grad_variable.clone();
            }
            None => {
                grad.set_name(&format!("{name}_grad"));
                *grad_mut = Some(grad);
            }
        }
    }

    pub fn get_is_train(&self) -> bool {
        self.inner.borrow().get_is_train()
    }

    pub fn set_is_train(&self, is_train: bool) {
        self.inner.borrow_mut().set_is_train(is_train);
    }

    pub fn get_all_trainable_variables(&self) -> Vec<Variable<T, D>> {
        self.inner.borrow().get_all_trainable_variables()
    }

    pub fn get_shape(&self) -> DimDyn {
        self.get_data().shape()
    }

    pub fn to<DO: Device>(&self) -> Variable<T, DO> {
        Variable {
            inner: Rc::new(RefCell::new(self.inner.borrow().to())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VariableWeak<T: Num, D: Device> {
    inner: Weak<RefCell<VariableInner<T, D>>>,
}

impl<T: Num, D: Device> VariableWeak<T, D> {
    pub fn upgrade(&self) -> Option<Variable<T, D>> {
        self.inner.upgrade().map(|inner| Variable { inner })
    }
}

impl<T: Num, D: Device> Debug for Variable<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.get_data();
        write!(f, "Variable {{ data: \n{:?} }}", inner)?;
        Ok(())
    }
}

impl<T: Num, D: Device> Display for Variable<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.get_data();
        write!(f, "Variable {{ data: \n{:?} }}", inner)?;
        Ok(())
    }
}

impl<T: Num, D: Device> Debug for VariableInner<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VariableInner {{ data: \n{:?} }}", self.data)?;
        Ok(())
    }
}

impl<T: Num, D: Device> Display for VariableInner<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VariableInner {{ data: \n{:?} }}", self.data)?;
        Ok(())
    }
}

use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashSet},
    fmt::Debug,
    ops::Deref,
    rc::{Rc, Weak},
};

#[derive(Debug)]
pub struct VariableInner {
    data: f64,
    creator: Option<Rc<RefCell<Box<dyn Function>>>>,
    grad: Option<Variable>,
    gen: usize,
    name: Option<String>,
}

impl VariableInner {
    pub fn new(data: f64) -> Self {
        Self {
            data,
            creator: None,
            grad: None,
            gen: 0,
            name: None,
        }
    }

    pub fn get_data(&self) -> f64 {
        self.data
    }

    pub fn set_data(&mut self, data: f64) {
        self.data = data;
    }

    pub fn set_creator(&mut self, creator: Rc<RefCell<Box<dyn Function>>>) {
        self.creator = Some(creator);
        let gen = self.creator.as_ref().unwrap().borrow().get_gen();
        self.gen = gen + 1;
    }

    pub fn get_creator(&self) -> &Option<Rc<RefCell<Box<dyn Function>>>> {
        &self.creator
    }

    pub fn get_grad(&self) -> &Option<Variable> {
        &self.grad
    }

    pub fn set_grad(&mut self, grad: Variable) {
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
        let mut funcs: BinaryHeap<FunctionQueueItem> = BinaryHeap::new();
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
        match self.grad {
            Some(ref mut grad) => {
                *grad = Variable::new(0.);
            }
            None => {}
        }
    }

    pub fn set_name(&mut self, name: &str) {
        self.name = Some(name.to_string());
    }

    pub fn get_name(&self) -> &Option<String> {
        &self.name
    }
}

#[derive(Debug, Clone)]
pub struct Variable {
    inner: Rc<RefCell<VariableInner>>,
}

impl Variable {
    pub fn new(data: f64) -> Self {
        let inner = VariableInner::new(data);
        let inner = Rc::new(RefCell::new(inner));
        Self { inner }
    }

    pub fn get_data(&self) -> f64 {
        self.inner.borrow().get_data()
    }

    pub fn set_data(&self, data: f64) {
        self.inner.borrow_mut().set_data(data);
    }

    pub fn set_creator(&self, creator: Rc<RefCell<Box<dyn Function>>>) {
        self.inner.borrow_mut().set_creator(creator);
    }

    pub fn get_creator(&self) -> Option<Rc<RefCell<Box<dyn Function>>>> {
        self.inner.borrow().get_creator().clone()
    }

    pub fn get_grad(&self) -> Option<Variable> {
        self.inner.borrow().get_grad().clone()
    }

    pub fn set_grad(&self, grad: Variable) {
        self.inner.borrow_mut().set_grad(grad);
    }

    pub fn backward(&self) {
        if self.inner.borrow().get_grad().is_none() {
            self.inner.borrow_mut().set_grad(Variable::new(1.));
        }
        self.inner.borrow().backward();
    }

    pub fn downgrade(self) -> VariableWeak {
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

impl AsRef<Variable> for Variable {
    fn as_ref(&self) -> &Variable {
        self
    }
}

#[derive(Debug, Clone)]
pub struct VariableWeak {
    inner: Weak<RefCell<VariableInner>>,
}

impl VariableWeak {
    pub fn upgrade(&self) -> Option<Variable> {
        self.inner.upgrade().map(|inner| Variable { inner })
    }
}

pub trait Function: Debug {
    fn forward(&self);
    fn backward(&self);
    fn get_inputs(&self) -> Vec<Variable>;
    fn get_gen(&self) -> usize {
        let inputs = self.get_inputs();
        inputs
            .iter()
            .map(|input| input.get_gen())
            .max()
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct FunctionQueueItem {
    func: Rc<RefCell<Box<dyn Function>>>,
    gen: usize,
}

impl From<Rc<RefCell<Box<dyn Function>>>> for FunctionQueueItem {
    fn from(func: Rc<RefCell<Box<dyn Function>>>) -> Self {
        Self {
            func: func.clone(),
            gen: func.borrow().get_gen(),
        }
    }
}

impl PartialEq for FunctionQueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.gen == other.gen
    }
}

impl Eq for FunctionQueueItem {
    fn assert_receiver_is_total_eq(&self) {}
}

impl PartialOrd for FunctionQueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.gen.cmp(&other.gen))
    }
}

impl Ord for FunctionQueueItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gen.cmp(&other.gen)
    }
}

impl Deref for FunctionQueueItem {
    type Target = Rc<RefCell<Box<dyn Function>>>;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

#[derive(Debug)]
pub struct Add {
    x: Variable,
    y: Variable,
    output: VariableWeak,
}

impl Add {
    pub fn new(x: Variable, y: Variable, output: Variable) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl Function for Add {
    fn forward(&self) {
        let ans = self.x.get_data() + self.y.get_data();
        self.output.upgrade().unwrap().set_data(ans);
    }

    fn backward(&self) {
        let input_grad = self.output.upgrade().unwrap().get_grad().unwrap();
        self.x.set_grad(input_grad.clone());
        self.y.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable> {
        vec![self.x.clone(), self.y.clone()]
    }
}

pub fn add<V: AsRef<Variable>>(x: V, y: V) -> Variable {
    let x = x.as_ref().clone();
    let y = y.as_ref().clone();
    let output = Variable::new(0.);
    let add = Add::new(x, y, output.clone());
    add.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(add))));
    output
}

#[derive(Debug)]
pub struct Mul {
    x: Variable,
    y: Variable,
    output: VariableWeak,
}

impl Mul {
    pub fn new(x: Variable, y: Variable, output: Variable) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl Function for Mul {
    fn forward(&self) {
        let ans = self.x.get_data() * self.y.get_data();
        self.output.upgrade().unwrap().set_data(ans);
    }

    fn backward(&self) {
        let input_grad = self.output.upgrade().unwrap().get_grad().unwrap();
        let x_grad = mul(&input_grad, &self.y);
        let y_grad = mul(&input_grad, &self.x);
        self.x.set_grad(x_grad);
        self.y.set_grad(y_grad);
    }

    fn get_inputs(&self) -> Vec<Variable> {
        vec![self.x.clone(), self.y.clone()]
    }
}

pub fn mul<V: AsRef<Variable>>(x: V, y: V) -> Variable {
    let x = x.as_ref().clone();
    let y = y.as_ref().clone();
    let output = Variable::new(0.);
    let mul = Mul::new(x, y, output.clone());
    mul.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(mul))));
    output
}

#[derive(Debug)]
pub struct Square {
    x: Variable,
    output: VariableWeak,
}

impl Square {
    pub fn new(x: Variable, output: Variable) -> Self {
        let output = output.downgrade();
        Self { x, output }
    }
}

impl Function for Square {
    fn forward(&self) {
        let ans = self.x.get_data() * self.x.get_data();
        self.output.upgrade().unwrap().set_data(ans);
    }

    fn backward(&self) {
        let grad = self.output.upgrade().unwrap().get_grad().unwrap();
        let tmp = mul(&grad, &Variable::new(2.));
        let x_grad = mul(&tmp, &self.x);
        self.x.set_grad(x_grad);
    }

    fn get_inputs(&self) -> Vec<Variable> {
        vec![self.x.clone()]
    }
}

pub fn square<V: AsRef<Variable>>(x: V) -> Variable {
    let output = Variable::new(0.);
    let square = Square::new(x.as_ref().clone(), output.clone());
    square.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(square))));
    output
}

#[cfg(test)]
mod autograd {
    use super::*;

    #[test]
    fn add_test() {
        let x0 = Variable::new(2.0);
        let x1 = Variable::new(3.0);
        let y = add(x0, x1);
        assert_eq!(y.get_data(), 5.0);
    }

    #[test]
    fn mul_test() {
        let x0 = Variable::new(2.0);
        let x1 = Variable::new(3.0);
        let y = mul(x0, x1);
        assert_eq!(y.get_data(), 6.0);
    }

    #[test]
    fn add_mul() {
        let x0 = Variable::new(2.0);
        let x1 = Variable::new(3.0);
        let x2 = Variable::new(4.0);
        let y = mul(add(x0, x1), x2);
        assert_eq!(y.get_data(), 20.0);
    }

    #[test]
    fn combined_backward() {
        let x0 = Variable::new(2.0);
        let x1 = Variable::new(3.0);
        let x2 = Variable::new(4.0);
        let y = add(&x0, &x1);
        let y = mul(y, x2.clone());
        y.backward();
        assert_eq!(x0.get_grad().unwrap().get_data(), 4.0);
        assert_eq!(x1.get_grad().unwrap().get_data(), 4.0);
        assert_eq!(x2.get_grad().unwrap().get_data(), 5.0);
    }

    #[test]
    fn use_same_variable() {
        let x0 = Variable::new(2.0);
        let y = add(x0.clone(), x0.clone());
        y.backward();
        assert_eq!(x0.get_grad().unwrap().get_data(), 2.0);
    }

    #[test]
    fn complicated_backward() {
        let x = Variable::new(2.0);
        let a = square(x.clone());
        let y = add(square(a.clone()), square(a.clone()));
        y.backward();
        assert_eq!(y.get_data(), 32.0);
        assert_eq!(x.get_grad().unwrap().get_data(), 64.0);
    }

    #[test]
    fn grad_twice() {
        let x = Variable::new(3.0);
        x.set_name("x");
        let y = square(x.clone());
        y.backward();
        assert_eq!(x.get_grad().unwrap().get_data(), 6.0);
        let x_grad = x.get_grad().unwrap();
        x.clear_grad();
        x_grad.backward();
        assert_eq!(x.get_grad().unwrap().get_data(), 2.0);
    }
}

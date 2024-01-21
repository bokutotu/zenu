use std::{
    cell::RefCell,
    fmt::Debug,
    rc::{Rc, Weak},
};

#[derive(Debug)]
pub struct VariableInner {
    data: f64,
    creator: Option<Rc<RefCell<Box<dyn Function>>>>,
    grad: Option<f64>,
}

impl VariableInner {
    pub fn new(data: f64) -> Self {
        Self {
            data,
            creator: None,
            grad: None,
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
    }

    pub fn get_creator(&self) -> &Option<Rc<RefCell<Box<dyn Function>>>> {
        &self.creator
    }

    pub fn get_grad(&self) -> &Option<f64> {
        &self.grad
    }

    pub fn set_grad(&mut self, grad: f64) {
        self.grad = Some(grad);
    }

    pub fn backward(&self) {
        let mut funcs = vec![self.creator.clone()];
        while let Some(func) = funcs.pop() {
            if let Some(func) = func {
                func.borrow().backward();
                let inputs = func.borrow().get_inputs();
                funcs.extend(inputs.into_iter().map(|input| input.get_creator()));
            }
        }
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

    pub fn get_grad(&self) -> Option<f64> {
        self.inner.borrow().get_grad().clone()
    }

    pub fn set_grad(&self, grad: f64) {
        self.inner.borrow_mut().set_grad(grad);
    }

    pub fn backward(&self) {
        if self.inner.borrow().get_grad().is_none() {
            self.inner.borrow_mut().set_grad(1.);
        }
        self.inner.borrow().backward();
    }

    pub fn downgrade(self) -> VariableWeak {
        VariableWeak {
            inner: Rc::downgrade(&self.inner),
        }
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
}

#[derive(Debug)]
pub struct Add {
    x: Variable,
    y: Variable,
    output: Variable,
}

impl Add {
    pub fn new(x: Variable, y: Variable, output: Variable) -> Self {
        Self { x, y, output }
    }
}

impl Function for Add {
    fn forward(&self) {
        let ans = self.x.get_data() + self.y.get_data();
        self.output.set_data(ans);
    }

    fn backward(&self) {
        let grad = self.output.get_grad().unwrap();
        self.x.set_grad(grad);
        self.y.set_grad(grad);
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
    output.clone()
}

#[derive(Debug)]
pub struct Mul {
    x: Variable,
    y: Variable,
    output: Variable,
}

impl Mul {
    pub fn new(x: Variable, y: Variable, output: Variable) -> Self {
        Self { x, y, output }
    }
}

impl Function for Mul {
    fn forward(&self) {
        let ans = self.x.get_data() * self.y.get_data();
        self.output.set_data(ans);
    }

    fn backward(&self) {
        let grad = self.output.get_grad().unwrap();
        println!("mul");
        self.x.set_grad(grad * self.y.get_data());
        self.y.set_grad(grad * self.x.get_data());
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
    output.clone()
}

#[derive(Debug)]
pub struct Square {
    x: Variable,
    output: Variable,
}

impl Square {
    pub fn new(x: Variable, output: Variable) -> Self {
        Self { x, output }
    }
}

impl Function for Square {
    fn forward(&self) {
        let ans = self.x.get_data() * self.x.get_data();
        self.output.set_data(ans);
    }

    fn backward(&self) {
        let grad = self.output.get_grad().unwrap();
        let x = self.x.get_data();
        self.x.set_grad(grad * 2. * x);
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
    output.clone()
}

#[derive(Debug)]
pub struct Exp {
    x: Variable,
    output: Variable,
}

impl Exp {
    pub fn new(x: Variable, output: Variable) -> Self {
        Self { x, output }
    }
}

impl Function for Exp {
    fn forward(&self) {
        let ans = self.x.get_data().exp();
        self.output.set_data(ans);
    }

    fn backward(&self) {
        let grad = self.output.get_grad().unwrap();
        let x = self.x.get_data();
        self.x.set_grad(grad * x.exp());
    }

    fn get_inputs(&self) -> Vec<Variable> {
        vec![self.x.clone()]
    }
}

pub fn exp<V: AsRef<Variable>>(x: V) -> Variable {
    let x = x.as_ref();
    let output = Variable::new(0.);
    let exp = Exp::new(x.clone(), output.clone());
    exp.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(exp))));
    output.clone()
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
        assert_eq!(x0.get_grad(), Some(4.0));
        assert_eq!(x1.get_grad(), Some(4.0));
        assert_eq!(x2.get_grad(), Some(5.0));
    }
}

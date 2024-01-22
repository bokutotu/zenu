use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashSet},
    fmt::Debug,
    ops::Deref,
    rc::{Rc, Weak},
};

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
    fn one() -> Self;
}

pub trait Value:
    Zero + One + Clone + Debug + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> + 'static
{
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0
    }
}

impl One for f64 {
    fn one() -> Self {
        1.0
    }
}

impl Value for f32 {}

impl Value for f64 {}

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
        match self.grad {
            Some(ref mut grad) => {
                *grad = Variable::new(V::zero());
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
pub struct FunctionQueueItem<V> {
    func: Rc<RefCell<Box<dyn Function<V>>>>,
    gen: usize,
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

pub struct Add<V> {
    x: Variable<V>,
    y: Variable<V>,
    output: VariableWeak<V>,
}

impl<V: Value> Add<V> {
    pub fn new(x: Variable<V>, y: Variable<V>, output: Variable<V>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<V: Value> Function<V> for Add<V> {
    fn forward(&self) {
        let ans = self.x.get_data() + self.y.get_data();
        self.output.upgrade().unwrap().set_data(ans);
    }

    fn backward(&self) {
        let input_grad = self.output.upgrade().unwrap().get_grad().unwrap();
        self.x.set_grad(input_grad.clone());
        self.y.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<V>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

pub fn add<V1: Value, V: AsRef<Variable<V1>>>(x: V, y: V) -> Variable<V1> {
    let x = x.as_ref().clone();
    let y = y.as_ref().clone();
    let output = Variable::new(V1::zero());
    let add = Add::new(x, y, output.clone());
    add.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(add))));
    output
}

#[cfg(test)]
mod autograd {
    use super::*;
    pub struct Mul<V> {
        x: Variable<V>,
        y: Variable<V>,
        output: VariableWeak<V>,
    }

    impl<V: Value> Mul<V> {
        pub fn new(x: Variable<V>, y: Variable<V>, output: Variable<V>) -> Self {
            let output = output.downgrade();
            Self { x, y, output }
        }
    }

    impl<V: Value> Function<V> for Mul<V> {
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

        fn get_inputs(&self) -> Vec<Variable<V>> {
            vec![self.x.clone(), self.y.clone()]
        }
    }

    pub fn mul<V1: Value, V: AsRef<Variable<V1>>>(x: V, y: V) -> Variable<V1> {
        let x = x.as_ref().clone();
        let y = y.as_ref().clone();
        let output = Variable::new(V1::zero());
        let mul = Mul::new(x, y, output.clone());
        mul.forward();
        output.set_creator(Rc::new(RefCell::new(Box::new(mul))));
        output
    }

    pub struct Square<V> {
        x: Variable<V>,
        output: VariableWeak<V>,
    }

    impl<V: Value> Square<V> {
        pub fn new(x: Variable<V>, output: Variable<V>) -> Self {
            let output = output.downgrade();
            Self { x, output }
        }
    }

    impl<V: Value> Function<V> for Square<V> {
        fn forward(&self) {
            let ans = self.x.get_data() * self.x.get_data();
            self.output.upgrade().unwrap().set_data(ans);
        }

        fn backward(&self) {
            let grad = self.output.upgrade().unwrap().get_grad().unwrap();
            let tmp = mul(&grad, &Variable::new(V::one() + V::one()));
            let x_grad = mul(&tmp, &self.x);
            self.x.set_grad(x_grad);
        }

        fn get_inputs(&self) -> Vec<Variable<V>> {
            vec![self.x.clone()]
        }
    }

    pub fn square<V1: Value, V: AsRef<Variable<V1>>>(x: V) -> Variable<V1> {
        let output = Variable::new(V1::zero());
        let square = Square::new(x.as_ref().clone(), output.clone());
        square.forward();
        output.set_creator(Rc::new(RefCell::new(Box::new(square))));
        output
    }
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

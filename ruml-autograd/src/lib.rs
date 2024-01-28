pub mod functions;
pub mod value;
pub mod variable;

#[cfg(test)]
mod autograd {
    use std::{cell::RefCell, rc::Rc};
    use value::Value;

    use crate::{
        functions::{add::add, Function},
        variable::{Variable, VariableWeak},
    };

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
        let output = Variable::new(V1::zero(&[]));
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
            let tmp = mul(&grad, &Variable::new(V::one(&[]) + V::one(&[])));
            let x_grad = mul(&tmp, &self.x);
            self.x.set_grad(x_grad);
        }

        fn get_inputs(&self) -> Vec<Variable<V>> {
            vec![self.x.clone()]
        }
    }

    pub fn square<V1: Value, V: AsRef<Variable<V1>>>(x: V) -> Variable<V1> {
        let output = Variable::new(V1::zero(&[]));
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

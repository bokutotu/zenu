use std::ops::Sub;

use zenu_matrix::num::Num;

use crate::Variable;

pub fn sub<T: Num>(x: Variable<T>, y: Variable<T>) -> Variable<T> {
    let y = y * Variable::from(T::minus_one());
    x + y
}

impl<T: Num> Sub<Variable<T>> for Variable<T> {
    type Output = Variable<T>;

    fn sub(self, rhs: Variable<T>) -> Self::Output {
        sub(self, rhs)
    }
}

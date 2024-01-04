use crate::matrix::ViewMutMatix;

pub trait Mul<Rhs, Lhs>: ViewMutMatix {
    fn mul(&mut self, rhs: &Rhs, lhs: &mut Lhs);
}

pub trait MatMul<Rhs, Lhs>: ViewMutMatix {
    fn mat_mul(&mut self, rhs: &Rhs, lhs: &mut Lhs);
}

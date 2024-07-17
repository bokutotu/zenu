use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{
    device::Device,
    dim::{larger_shape, DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref, Repr},
    num::Num,
};

macro_rules! call_on_self {
    ($self:ident, $F:ident, $($args:expr),*) => {
        $self.$F($($args),*)
    };
}

macro_rules! impl_arithmetic_ops {
    (
        $trait:ident,
        $trait_method:ident,
        $assign_trait:ident,
        $assign_trait_method:ident,
        $scalr:ident,
        $scalar_assign:ident,
        $array:ident,
        $array_assign:ident
    ) => {
        // Add<T> for Matrix<R, S, D>
        impl<T: Num, R: Repr<Item = T>, S: DimTrait, D: Device> $trait<T> for Matrix<R, S, D> {
            type Output = Matrix<Owned<T>, S, D>;

            fn $trait_method(self, rhs: T) -> Self::Output {
                let mut owned = Matrix::alloc_like(&self);
                {
                    let mut ref_mut = owned.to_ref_mut();
                    call_on_self!(ref_mut, $scalr, &self, rhs);
                }
                owned
            }
        }

        impl<T: Num, R: Repr<Item = T>, S: DimTrait, D: Device> $trait<T> for &Matrix<R, S, D> {
            type Output = Matrix<Owned<T>, S, D>;

            fn $trait_method(self, rhs: T) -> Self::Output {
                let mut owned = Matrix::alloc_like(self);
                {
                    let mut ref_mut = owned.to_ref_mut();
                    call_on_self!(ref_mut, $scalr, self, rhs);
                }
                owned
            }
        }

        // Add<Matrix<RO, SO, D>> for Matrix<RS, SS, D>
        impl<
                T: Num,
                RS: Repr<Item = T>,
                SS: DimTrait,
                RO: Repr<Item = T>,
                SO: DimTrait,
                D: Device,
            > $trait<Matrix<RO, SO, D>> for Matrix<RS, SS, D>
        {
            type Output = Matrix<Owned<T>, DimDyn, D>;

            fn $trait_method(self, rhs: Matrix<RO, SO, D>) -> Self::Output {
                let larger = if self.shape().len() == rhs.shape().len() {
                    DimDyn::from(larger_shape(self.shape(), rhs.shape()))
                } else if self.shape().len() > rhs.shape().len() {
                    DimDyn::from(self.shape().slice())
                } else {
                    DimDyn::from(rhs.shape().slice())
                };
                let mut owned: Matrix<Owned<T>, DimDyn, D> = Matrix::alloc(larger.slice());
                {
                    let mut ref_mut = owned.to_ref_mut();
                    call_on_self!(ref_mut, $array, &self, &rhs);
                }
                owned
            }
        }

        impl<
                T: Num,
                RS: Repr<Item = T>,
                SS: DimTrait,
                RO: Repr<Item = T>,
                SO: DimTrait,
                D: Device,
            > $trait<&Matrix<RO, SO, D>> for Matrix<RS, SS, D>
        {
            type Output = Matrix<Owned<T>, DimDyn, D>;

            fn $trait_method(self, rhs: &Matrix<RO, SO, D>) -> Self::Output {
                let larger = if self.shape().len() == rhs.shape().len() {
                    DimDyn::from(larger_shape(self.shape(), rhs.shape()))
                } else if self.shape().len() > rhs.shape().len() {
                    DimDyn::from(self.shape().slice())
                } else {
                    DimDyn::from(rhs.shape().slice())
                };
                let mut owned: Matrix<Owned<T>, DimDyn, D> = Matrix::alloc(larger.slice());
                {
                    let mut ref_mut = owned.to_ref_mut();
                    call_on_self!(ref_mut, $array, &self, &rhs);
                }
                owned
            }
        }
        impl<
                T: Num,
                RS: Repr<Item = T>,
                SS: DimTrait,
                RO: Repr<Item = T>,
                SO: DimTrait,
                D: Device,
            > $trait<Matrix<RO, SO, D>> for &Matrix<RS, SS, D>
        {
            type Output = Matrix<Owned<T>, DimDyn, D>;

            fn $trait_method(self, rhs: Matrix<RO, SO, D>) -> Self::Output {
                let larger = if self.shape().len() == rhs.shape().len() {
                    DimDyn::from(larger_shape(self.shape(), rhs.shape()))
                } else if self.shape().len() > rhs.shape().len() {
                    DimDyn::from(self.shape().slice())
                } else {
                    DimDyn::from(rhs.shape().slice())
                };
                let mut owned: Matrix<Owned<T>, DimDyn, D> = Matrix::alloc(larger.slice());
                {
                    let mut ref_mut = owned.to_ref_mut();
                    call_on_self!(ref_mut, $array, &self, &rhs);
                }
                owned
            }
        }

        impl<
                T: Num,
                RS: Repr<Item = T>,
                SS: DimTrait,
                RO: Repr<Item = T>,
                SO: DimTrait,
                D: Device,
            > $trait<&Matrix<RO, SO, D>> for &Matrix<RS, SS, D>
        {
            type Output = Matrix<Owned<T>, DimDyn, D>;

            fn $trait_method(self, rhs: &Matrix<RO, SO, D>) -> Self::Output {
                let larger = if self.shape().len() == rhs.shape().len() {
                    DimDyn::from(larger_shape(self.shape(), rhs.shape()))
                } else if self.shape().len() > rhs.shape().len() {
                    DimDyn::from(self.shape().slice())
                } else {
                    DimDyn::from(rhs.shape().slice())
                };
                let mut owned: Matrix<Owned<T>, DimDyn, D> = Matrix::alloc(larger.slice());
                {
                    let mut ref_mut = owned.to_ref_mut();
                    call_on_self!(ref_mut, $array, &self, &rhs);
                }
                owned
            }
        }

        // AddAssign<T> for Matrix<Ref<&mut T>, S, D>
        impl<T: Num, S: DimTrait, D: Device> $assign_trait<T> for Matrix<Ref<&mut T>, S, D> {
            fn $assign_trait_method(&mut self, rhs: T) {
                call_on_self!(self, $scalar_assign, rhs);
            }
        }

        // AddAssign<T> for Matrix<Owned<T>, S, D>
        impl<T: Num, S: DimTrait, D: Device> $assign_trait<T> for Matrix<Owned<T>, S, D> {
            fn $assign_trait_method(&mut self, rhs: T) {
                let mut ref_mut = self.to_ref_mut();
                call_on_self!(ref_mut, $scalar_assign, rhs);
            }
        }

        // AddAssign<Matrix<RO, SO, D>> for Matrix<Owned<T>, SS, D>
        impl<T: Num, SS: DimTrait, RO: Repr<Item = T>, SO: DimTrait, D: Device>
            $assign_trait<Matrix<RO, SO, D>> for Matrix<Owned<T>, SS, D>
        {
            fn $assign_trait_method(&mut self, rhs: Matrix<RO, SO, D>) {
                let mut ref_mut = self.to_ref_mut();
                call_on_self!(ref_mut, $array_assign, &rhs);
            }
        }

        // AddAssign<Matrix<R, SO, D>> for Matrix<Ref<&mut T>, SS, D>
        impl<T: Num, R: Repr<Item = T>, SO: DimTrait, SS: DimTrait, D: Device>
            $assign_trait<Matrix<R, SO, D>> for Matrix<Ref<&mut T>, SS, D>
        {
            fn $assign_trait_method(&mut self, rhs: Matrix<R, SO, D>) {
                call_on_self!(self, $array_assign, &rhs);
            }
        }
    };
}
impl_arithmetic_ops!(
    Add,
    add,
    AddAssign,
    add_assign,
    add_scalar,
    add_scalar_assign,
    add_array,
    add_assign
);
impl_arithmetic_ops!(
    Sub,
    sub,
    SubAssign,
    sub_assign,
    sub_scalar,
    sub_scalar_assign,
    sub_array,
    sub_assign
);
impl_arithmetic_ops!(
    Mul,
    mul,
    MulAssign,
    mul_assign,
    mul_scalar,
    mul_scalar_assign,
    mul_array,
    mul_assign
);
impl_arithmetic_ops!(
    Div,
    div,
    DivAssign,
    div_assign,
    div_scalar,
    div_scalar_assign,
    div_array,
    div_assign
);

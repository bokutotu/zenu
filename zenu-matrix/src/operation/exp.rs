use crate::{
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{IndexAxisDyn, IndexAxisMutDyn, MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::{ToViewMemory, ToViewMutMemory},
    memory_impl::OwnedMem,
    num::Num,
};

use super::zeros::Zeros;

pub trait Exp<T: Num> {
    fn exp(&self) -> Matrix<OwnedMem<T>, DimDyn>;
}

impl<T: Num, M: ToViewMemory<Item = T>> Exp<T> for Matrix<M, DimDyn> {
    fn exp(&self) -> Matrix<OwnedMem<T>, DimDyn> {
        let mut owned = Matrix::<OwnedMem<T>, DimDyn>::zeros_like(self.to_view());
        let v = self.to_view();
        owned.to_view_mut().exp_assign(&v);
        owned
    }
}

pub trait ExpAssign<T: Num> {
    fn exp_assign<V: ToViewMemory<Item = T>>(&mut self, y: &Matrix<V, DimDyn>);
}

impl<T: Num, M: ToViewMutMemory<Item = T>> ExpAssign<T> for Matrix<M, DimDyn> {
    fn exp_assign<V: ToViewMemory<Item = T>>(&mut self, y: &Matrix<V, DimDyn>) {
        let y = y.to_view();
        assert_eq!(self.shape(), y.shape());
        let len = self.shape().len();
        if len <= 1 {
            let incs = if len == 0 { 1 } else { self.stride()[0] };
            let incx = if len == 0 { 1 } else { y.stride()[0] };
            let num_elm = if len == 0 { 1 } else { self.shape()[0] };
            exp_kernel_cpu(self.as_mut_slice(), y.as_slice(), num_elm, incs, incx);
        } else {
            for i in 0..self.shape()[0] {
                self.to_view_mut()
                    .index_axis_mut_dyn(Index0D::new(i))
                    .exp_assign(&y.index_axis_dyn(Index0D::new(i)));
            }
        }
    }
}

fn exp_kernel_cpu<T: Num>(x: &mut [T], y: &[T], len: usize, incx: usize, incy: usize) {
    if incx == 1 && incy == 1 {
        for i in 0..len {
            x[i] = y[i].exp();
        }
    } else {
        for i in 0..len {
            x[i * incx] = y[i * incy].exp();
        }
    }
}

#[cfg(test)]
mod exp {
    use crate::{
        matrix::{OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::{asum::Asum, zeros::Zeros},
    };

    use super::ExpAssign;

    #[test]
    fn exp_0d() {
        let x = OwnedMatrixDyn::from_vec(vec![2.], &[]);
        let mut y = OwnedMatrixDyn::zeros(&[]);
        y.to_view_mut().exp_assign(&x);

        let diff: f64 = y.to_view().as_slice()[0] - 7.3890560989306495;
        let diff: f64 = diff.abs();
        assert!(diff < 1e-10);
    }

    #[test]
    fn exp_2d() {
        let x = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4.], &[2, 2]);
        let mut y = OwnedMatrixDyn::zeros(&[2, 2]);
        y.to_view_mut().exp_assign(&x);
        let ans = OwnedMatrixDyn::from_vec(
            vec![
                2.718281828459045,
                7.3890560989306495,
                20.085536923187668,
                54.598150033144236,
            ],
            &[2, 2],
        );
        let diff = y.to_view() - ans.to_view();
        let diff = diff.asum();
        assert!(diff < 1e-10);
    }
}

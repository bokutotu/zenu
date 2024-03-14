use crate::{
    dim::DimTrait,
    matrix::{IndexItemAsign, MatrixBase, ToViewMatrix},
    matrix_blas::dot::dot as dot_func,
    matrix_impl::{matrix_into_dim, Matrix},
    memory::{ToViewMemory, ViewMut},
    num::Num,
};

pub trait Dot<RHS, LHS> {
    /// Compute the dot product of two vectors.
    /// - If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
    /// - If both a and b are 2-D arrays, it is matrix multiplication, but using `matmul``.
    /// - If either a or b is 0-D (scalar),
    ///   it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
    /// - If a is an N-D array and b is a 1-D array, it is a sum product over t
    /// - If a is an N-D array and b is an M-D array (where M>=2),
    ///   it is a sum product over the last axis of a and the second-to-last axis of b:he last axis of a and b.
    fn dot(self, rhs: RHS, lhs: LHS);
}

impl<T, SM, RM, LM, DS, DR, DL> Dot<Matrix<RM, DR>, Matrix<LM, DL>> for Matrix<SM, DS>
where
    T: Num,
    SM: ViewMut<Item = T>,
    RM: ToViewMemory<Item = T>,
    LM: ToViewMemory<Item = T>,
    DS: DimTrait,
    DR: DimTrait,
    DL: DimTrait,
{
    fn dot(self, rhs: Matrix<RM, DR>, lhs: Matrix<LM, DL>) {
        dot_shape_check(self.shape(), rhs.shape(), lhs.shape());
        let result = dot_func(
            matrix_into_dim(rhs).to_view(),
            matrix_into_dim(lhs).to_view(),
        );
        let mut s = self.into_dyn_dim();
        s.index_item_asign(&[], result);
    }
}

fn dot_shape_check<SD, RD, LD>(self_shape: SD, rhs_shape: RD, lhs_shape: LD)
where
    SD: DimTrait,
    RD: DimTrait,
    LD: DimTrait,
{
    if rhs_shape.len() == 1
        && lhs_shape.len() == 1
        && self_shape.len() == 0
        && rhs_shape[0] == lhs_shape[0]
    {
        return;
    }
    if rhs_shape.len() < lhs_shape.len() {
        dot_shape_check(self_shape, lhs_shape, rhs_shape);
        return;
    }
    if rhs_shape.len() > 2 {
        panic!("dot only supports 1-D and 2-D arrays");
    }
    // if rhs_shape and lhs_shape's length is all 2 gemm. so gemm_shape check
}

#[cfg(test)]
mod dot {
    use crate::{
        matrix::{IndexItem, OwnedMatrix, ToViewMutMatrix},
        matrix_impl::{OwnedMatrix0D, OwnedMatrix1D},
    };

    use super::Dot;

    #[test]
    fn dot() {
        let a = OwnedMatrix1D::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let b = OwnedMatrix1D::from_vec(vec![4.0, 5.0, 6.0], [3]);
        let mut c = OwnedMatrix0D::from_vec(vec![0.0], []);
        c.to_view_mut().dot(a, b);

        assert_eq!(c.index_item([]), 32.0);
    }
}

use crate::{
    dim::DimTrait,
    index::Index0D,
    matrix::{
        IndexAxisDyn, IndexAxisMutDyn, IndexItemAsign, MatrixBase, ToViewMatrix, ToViewMutMatrix,
    },
    matrix_blas::{dot::dot, gemm::gemm, gemv::gemv},
    matrix_impl::{matrix_into_dim, Matrix},
    memory::{ToViewMemory, ToViewMutMemory, ViewMut},
    num::Num,
};

pub trait Gemm<Rhs, Lhs>: ToViewMutMatrix {
    fn gemm(self, rhs: Rhs, lhs: Lhs);
    fn gemv(self, rhs: Rhs, lhs: Lhs, alpha: Self::Item, beta: Self::Item);
    fn gemm_batch(self, rhs: Rhs, lhs: Lhs);
    fn gemv_batch(self, rhs: Rhs, lhs: Lhs);
    /// English:
    ///- If both tensors are 1-dimensional, the dot product (scalar) is returned.
    // - If both arguments are 2-dimensional, a matrix-matrix product is returned.
    // - If the first argument is 1-dimensional and the second argument is 2-dimensional,
    //   a 1 is prepended to its dimension for the purpose of matrix multiplication.
    //   After matrix multiplication, the added dimension is removed.
    // - If the first argument is 2-dimensional and the second argument is 1-dimensional,
    //   a matrix-vector product is returned.
    // - If both arguments are at least 1-dimensional and at least one argument is N-dimensional (N > 2),
    //   a batched matrix multiplication is returned.
    //   If the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose
    //   of batched matrix multiplication, and then removed. If the second argument is 1-dimensional,
    //   a 1 is added to its dimension for the purpose of batched matrix multiplication,
    //   and then removed. Non-matrix (i.e., batch) dimensions must be broadcastable
    //   (thus, they must be broadcastable). For example, if one input is a (j × 1 × n × n) tensor
    //   and the other is a (k × n × n) tensor, the output will be a (j × k × n × n) tensor.
    // - The logic of broadcasting only looks at the batch dimensions to determine whether the inputs
    //   are broadcastable, but does not look at the matrix dimensions.
    //   For example, if one input is a (j × 1 × n × m) tensor and the other is a (k × m × p) tensor,
    //   these inputs are broadcastable, but the final 2 dimensions (i.e., the matrix dimensions) are different.
    //   The output will be a (j × k × n × p) tensor.
    //
    //  Japanese:
    /// - 両方のテンソルが1次元の場合、ドット積（スカラー）が返されます。
    /// - 両方の引数が2次元の場合、行列-行列積が返されます。
    /// - 第一引数が1次元で第二引数が2次元の場合、行列乗算の目的でその次元に1が先行付加されます。
    ///   行列乗算後、付加された次元は削除されます。
    /// - 第一引数が2次元で第二引数が1次元の場合、行列-ベクトル積が返されます。
    /// - 両方の引数が少なくとも1次元以上で、少なくとも一方の引数がN次元（N > 2）の場合、
    ///   バッチ化された行列乗算が返されます。第一引数が1次元の場合、
    ///   バッチ化された行列乗算の目的でその次元に1が先行付加され、その後削除されます。
    ///   第二引数が1次元の場合、バッチ化された行列乗算の目的でその次元に1が付加され、
    ///   その後削除されます。非行列（すなわちバッチ）次元はブロードキャストされる必要があります
    ///   （従って、ブロードキャスト可能でなければなりません）。
    ///   例えば、入力が(j × 1 × n × n)テンソルで、もう一方が(k × n × n)テンソルの場合、
    ///   出力は(j × k × n × n)テンソルになります。
    /// - ブロードキャストのロジックは、入力がブロードキャスト可能かどうかを判断する際に、
    ///   バッチ次元のみを見ますが、行列の次元は見ません。
    ///   例えば、入力が(j × 1 × n × m)テンソルで、もう一方が(k × m × p)テンソルの場合、
    ///   これらの入力はブロードキャスト可能ですが、最終的な2次元（すなわち行列次元）が異なります。
    ///   出力は(j × k × n × p)テンソルになります。
    /// - この操作は、スパースレイアウトの引数をサポートしています。
    ///   特に行列-行列積（両方の引数が2次元）は、torch.mm()と同じ制限でスパース引数をサポートします。
    fn matmul(self, rhs: Rhs, lhs: Lhs);
}

impl<'a, 'b, 'c, T, M1, M2, M3, D1, D2, D3> Gemm<Matrix<M1, D1>, Matrix<M2, D2>> for Matrix<M3, D3>
where
    T: Num,
    D1: DimTrait,
    D2: DimTrait,
    D3: DimTrait,
    M1: ToViewMemory<Item = T>,
    M2: ToViewMemory<Item = T>,
    M3: ViewMut<Item = T>,
{
    fn gemm(self, rhs: Matrix<M1, D1>, lhs: Matrix<M2, D2>) {
        assert_eq!(self.shape().len(), 2);
        assert_eq!(rhs.shape().len(), 2);
        assert_eq!(lhs.shape().len(), 2);
        let self_ = matrix_into_dim(self);
        let rhs = matrix_into_dim(rhs);
        let lhs = matrix_into_dim(lhs);
        gemm(rhs.to_view(), lhs.to_view(), self_, T::one(), T::zero());
    }

    fn gemv(self, rhs: Matrix<M1, D1>, lhs: Matrix<M2, D2>, alpha: T, beta: T) {
        assert_eq!(self.shape().len(), 1);
        assert_eq!(rhs.shape().len(), 2);
        assert_eq!(lhs.shape().len(), 1);
        let rhs = matrix_into_dim(rhs);
        let lhs = matrix_into_dim(lhs);
        let s = matrix_into_dim(self);
        gemv(rhs.to_view(), lhs.to_view(), s, alpha, beta);
    }

    fn gemm_batch(self, rhs: Matrix<M1, D1>, lhs: Matrix<M2, D2>) {
        assert!(self.shape().len() >= 2);
        assert!(rhs.shape().len() >= 2);
        assert!(lhs.shape().len() >= 2);
        if (self.shape().slice() != rhs.shape().slice())
            || (self.shape().slice() != lhs.shape().slice())
        {
            panic!("Dimension mismatch");
        }

        if self.shape().len() == 2 && rhs.shape().len() == 2 && lhs.shape().len() == 2 {
            self.gemm(rhs, lhs);
        } else {
            let self_shape_len = self.shape().len();
            let rhs_shape_len = rhs.shape().len();
            let lhs_shape_len = lhs.shape().len();

            let mut s = self.into_dyn_dim();
            let r = rhs.into_dyn_dim();
            let l = lhs.into_dyn_dim();

            for idx in 0..s.shape()[0] {
                let s = s.index_axis_mut_dyn(Index0D::new(idx));
                let r = if rhs_shape_len == self_shape_len {
                    r.index_axis_dyn(Index0D::new(idx))
                } else {
                    r.to_view()
                };
                let l = if lhs_shape_len == self_shape_len {
                    l.index_axis_dyn(Index0D::new(idx))
                } else {
                    l.to_view()
                };
                s.gemm(r, l);
            }
        }
    }

    fn gemv_batch(self, rhs: Matrix<M1, D1>, lhs: Matrix<M2, D2>) {
        todo!();
    }

    fn matmul(self, rhs: Matrix<M1, D1>, lhs: Matrix<M2, D2>) {
        todo!();
    }
}

#[cfg(test)]
mod mat_mul {
    use crate::{
        matrix::{IndexItem, OwnedMatrix, ToViewMatrix, ToViewMutMatrix},
        matrix_impl::OwnedMatrix2D,
        operation::zeros::Zeros,
    };

    use super::*;

    #[test]
    fn default() {
        let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let b = OwnedMatrix2D::from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
            ],
            [3, 5],
        );
        let mut ans = OwnedMatrix2D::<f32>::zeros([2, 5]);

        ans.to_view_mut().gemm(a.to_view(), b.to_view());
        assert_eq!(ans.index_item([0, 0]), 46.);
        assert_eq!(ans.index_item([0, 1]), 52.);
        assert_eq!(ans.index_item([0, 2]), 58.);
        assert_eq!(ans.index_item([0, 3]), 64.);
        assert_eq!(ans.index_item([0, 4]), 70.);
        assert_eq!(ans.index_item([1, 0]), 100.);
        assert_eq!(ans.index_item([1, 1]), 115.);
        assert_eq!(ans.index_item([1, 2]), 130.);
        assert_eq!(ans.index_item([1, 3]), 145.);
        assert_eq!(ans.index_item([1, 4]), 160.);
    }

    #[test]
    fn default_stride_2() {
        let a = OwnedMatrix2D::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        // shape 3 4
        let b = OwnedMatrix2D::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [3, 4],
        );
        let mut ans = OwnedMatrix2D::<f32>::zeros([2, 4]);

        ans.to_view_mut().gemm(a.to_view(), b.to_view());

        assert_eq!(ans.index_item([0, 0]), 38.);
        assert_eq!(ans.index_item([0, 1]), 44.);
        assert_eq!(ans.index_item([0, 2]), 50.);
        assert_eq!(ans.index_item([0, 3]), 56.);
        assert_eq!(ans.index_item([1, 0]), 83.);
        assert_eq!(ans.index_item([1, 1]), 98.);
        assert_eq!(ans.index_item([1, 2]), 113.);
        assert_eq!(ans.index_item([1, 3]), 128.);
    }
}

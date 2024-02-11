use crate::{
    blas::Blas,
    dim::{Dim1, Dim2, Dim3, Dim4, DimTrait},
    index::Index0D,
    matrix::{AsPtr, BlasMatrix, IndexAxis, IndexItem, MatrixBase, ViewMatrix},
    matrix_impl::Matrix,
    memory::{Memory, ViewMemory},
};

pub trait MaxIdx<D> {
    fn max_idx(&self) -> D;
}

impl<M: ViewMatrix + MatrixBase<Dim = Dim1>> MaxIdx<Dim1> for M {
    fn max_idx(&self) -> Dim1 {
        let idx = <M as BlasMatrix>::Blas::amax(
            self.shape_stride().shape().num_elm(),
            self.as_ptr(),
            self.shape_stride().stride()[0],
        );
        Dim1::new([idx])
    }
}

macro_rules! impl_max_idx {
    ($dim:ty) => {
        impl<M> MaxIdx<$dim> for Matrix<M, $dim>
        where
            M: ViewMemory,
        {
            fn max_idx(&self) -> $dim {
                if self.shape_stride().is_contiguous() {
                    let idx = <M as Memory>::Blas::amax(
                        self.shape_stride().shape().num_elm(),
                        self.as_ptr(),
                        self.shape_stride().stride()[self.shape_stride().shape().len() - 1],
                    );
                    self.shape_stride().get_dim_by_offset(idx)
                } else {
                    let mut max = self.shape_stride().get_dim_by_offset(0);
                    let mut max_val = self.index_item(max);
                    for i in 0..self.shape_stride().shape()[0] {
                        let tmp_matrix = self.index_axis(Index0D::new(i));
                        let tmp_max_dim = tmp_matrix.max_idx();
                        let tmp_max_val = tmp_matrix.index_item(tmp_max_dim);
                        if tmp_max_val > max_val {
                            max_val = tmp_max_val;
                            max[0] = i;
                            for j in 1..max.len() {
                                max[j] = tmp_max_dim[j - 1];
                            }
                        }
                    }
                    max
                }
            }
        }
    };
}
impl_max_idx!(Dim2);
impl_max_idx!(Dim3);
impl_max_idx!(Dim4);

#[cfg(test)]
mod max_idx {
    use crate::{
        dim,
        matrix::{MatrixSlice, OwnedMatrix, ToViewMatrix},
        matrix_impl::{CpuOwnedMatrix1D, CpuOwnedMatrix2D, CpuOwnedMatrix3D},
        operation::max::MaxIdx,
        slice,
    };

    #[test]
    fn default_1d() {
        let a = CpuOwnedMatrix1D::from_vec(vec![0., 1., 2., 3.], dim!(4));
        assert_eq!(a.to_view().max_idx(), dim!(3));
    }

    #[test]
    fn default_2d() {
        let a = CpuOwnedMatrix2D::from_vec(vec![0., 1., 2., 3.], dim!(2, 2));
        assert_eq!(a.to_view().max_idx(), dim!(1, 1));
    }

    #[test]
    fn sliced_3d() {
        let mut v = Vec::new();
        for i in 0..8 * 8 * 8 {
            v.push(i as f32);
        }
        let a = CpuOwnedMatrix3D::from_vec(v, dim!(8, 8, 8));
        let sliced = a.slice(slice!(..;3, ..;4, ..;2));
        assert_eq!(sliced.max_idx(), dim!(2, 1, 3));
    }
}

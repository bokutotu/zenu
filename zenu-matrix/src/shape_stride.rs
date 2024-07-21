use std::fmt::Debug;

use crate::dim::{default_stride, into_dyn, DimDyn, DimTrait};

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct ShapeStride<D: DimTrait> {
    shape: D,
    stride: D,
}

impl<D: DimTrait> ShapeStride<D> {
    pub fn new(shape: D, stride: D) -> Self {
        Self { shape, stride }
    }

    pub fn shape(&self) -> D {
        self.shape
    }

    pub fn stride(&self) -> D {
        self.stride
    }

    pub fn sort_by_stride(&self) -> Self {
        let mut indeies = (0..self.stride.len()).collect::<Vec<_>>();
        indeies.sort_by(|&a, &b| self.stride[b].cmp(&self.stride[a]));

        let shape = indeies.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();
        let stride = indeies.iter().map(|&i| self.stride[i]).collect::<Vec<_>>();

        let mut new_shape = self.shape();
        let mut new_stride = self.stride();

        for i in 0..self.stride.len() {
            new_shape[i] = shape[i];
            new_stride[i] = stride[i];
        }

        Self::new(new_shape, new_stride)
    }

    pub fn min_stride(&self) -> usize {
        let slice = self.stride.slice();
        *slice.iter().min().unwrap()
    }

    /// このShapeStrideが連続しているかどうかを判定する
    /// transposeされていた場合は並び替えを行い、
    /// そのストライドが、default_strideのn倍になっているかどうかを判定する
    pub fn is_contiguous(&self) -> bool {
        let sorted = self.sort_by_stride();

        let default_stride = default_stride(sorted.shape());

        let n = default_stride[0] / sorted.stride[0];

        let is_zero = default_stride[0] % sorted.stride[0] == 0;
        if !is_zero {
            return false;
        }

        let mut default_stride = default_stride;
        for i in 0..default_stride.len() {
            default_stride[i] *= n;
        }

        default_stride == sorted.stride
    }

    /// 転置は最後の次元と最後から2番目の次元を入れ替えることで表現される
    pub fn is_transposed(&self) -> bool {
        let last = self.stride()[self.stride().len() - 1];
        let last_2 = self.stride()[self.stride().len() - 2];

        last > last_2
    }

    pub fn transpose(&self) -> Self {
        let mut shape = self.shape();
        let mut stride = self.stride();

        let num_dim = shape.len();

        // 入れ替える
        let last = shape[shape.len() - 1];
        let last_2 = shape[shape.len() - 2];

        shape[num_dim - 1] = last_2;
        shape[num_dim - 2] = last;

        let last = stride[stride.len() - 1];
        let last_2 = stride[stride.len() - 2];

        stride[num_dim - 1] = last_2;
        stride[num_dim - 2] = last;

        Self::new(shape, stride)
    }

    pub fn is_default_stride(&self) -> bool {
        if self.shape().len() == 1 {
            return true;
        }
        default_stride(self.shape()) == self.stride()
    }

    /// shpae strideが転置されている場合、
    /// 転置を元に戻した場合default_strideになっているかどうかを判定する
    pub fn is_transposed_default_stride(&self) -> bool {
        self.transpose().is_default_stride()
    }

    pub(crate) fn into_dyn(self) -> ShapeStride<DimDyn> {
        let shape = into_dyn(self.shape);
        let stride = into_dyn(self.stride);
        ShapeStride::new(shape, stride)
    }

    pub(crate) fn transpose_by_index(&self, index: &[usize]) -> Self {
        let mut shape = self.shape();
        let mut stride = self.stride();

        let num_dim = shape.len();

        for i in 0..num_dim {
            shape[i] = self.shape()[index[i]];
            stride[i] = self.stride()[index[i]];
        }

        Self::new(shape, stride)
    }

    pub(crate) fn swap_index(self, a: usize, b: usize) -> Self {
        if a == b {
            return self;
        }
        if a >= self.shape().len() || b >= self.shape().len() {
            panic!("Index out of bounds");
        }
        let mut shape = self.shape();
        let mut stride = self.stride();

        let tmp_shape = shape[a];
        let tmp_stride = stride[a];

        shape[a] = shape[b];
        stride[a] = stride[b];

        shape[b] = tmp_shape;
        stride[b] = tmp_stride;

        Self::new(shape, stride)
    }
}

impl ShapeStride<DimDyn> {
    pub fn get_dim_by_offset(&self, offset: usize) -> DimDyn {
        let mut offset = offset;
        let mut dim = DimDyn::default();
        for i in 0..self.shape.len() {
            dim.push_dim(offset / self.stride[i]);
            offset %= self.stride[i];
        }
        dim
    }

    pub fn add_axis(self, axis: usize) -> Self {
        if self.shape().is_empty() {
            return ShapeStride::new(DimDyn::from([1]), DimDyn::from([1]));
        }
        let mut shape: DimDyn = Default::default();
        let mut stride: DimDyn = Default::default();

        for i in 0..self.shape.len() {
            if i == axis {
                shape.push_dim(1);
                stride.push_dim(self.stride[i]);
            }
            shape.push_dim(self.shape[i]);
            stride.push_dim(self.stride[i]);
        }
        if axis == self.shape.len() {
            shape.push_dim(1);
            stride.push_dim(1);
        }
        ShapeStride::new(shape, stride)
    }
}

#[cfg(test)]
mod shape_stride {
    use super::*;
    use crate::dim::{default_stride, Dim2, Dim4};

    #[test]
    fn is_transposed_false() {
        let shape = [2, 3];
        let shape: Dim2 = shape.into();
        let default_stride = default_stride(shape);

        let shape_stride = super::ShapeStride::new(shape, default_stride);

        assert!(!shape_stride.is_transposed());
    }

    #[test]
    fn is_transposed_true() {
        // transpose
        let shape_transposed = [2, 3, 5, 4];
        let stride_transposed = [60, 20, 1, 5];
        let shape_transposed: Dim4 = shape_transposed.into();
        let stride_transposed: Dim4 = stride_transposed.into();
        let shape_stride = ShapeStride::new(shape_transposed, stride_transposed);

        assert_eq!(shape_stride.is_transposed(), true);
    }
}

use std::fmt::Debug;

use crate::dim::{default_stride, DimTrait};

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

    pub fn is_transposed(&self) -> bool {
        println!("shape: {:?}", self.shape().into_iter().collect::<Vec<_>>());
        println!(
            "stride: {:?}",
            self.stride().into_iter().collect::<Vec<_>>()
        );
        let sorted = self.sort_by_stride();

        println!(
            "sorted shape: {:?}",
            sorted.shape().into_iter().collect::<Vec<_>>()
        );
        println!(
            "sorted stride: {:?}",
            sorted.stride().into_iter().collect::<Vec<_>>()
        );

        // sortedのshaptとself.shapeが一致しているかどうか
        // 一致していれば、transposeされていない
        sorted.shape != self.shape
    }
}

pub trait SliceTrait {
    type Dim: DimTrait;
    fn sliced_shape_stride(&self, shape: Self::Dim, stride: Self::Dim) -> ShapeStride<Self::Dim>;
    fn sliced_offset(&self, stride: Self::Dim, original_offset: usize) -> usize;
}

/// Matrixに対して、Indexを取得してTを取得するのに使用するトレイト
pub trait IndexTrait {
    type Dim: DimTrait;
    fn offset(&self, shape: &Self::Dim, stride: &Self::Dim) -> usize;
}

pub trait IndexAxisTrait {
    fn get_shape_stride<Din: DimTrait, Dout: DimTrait>(
        &self,
        shape: Din,
        stride: Din,
    ) -> ShapeStride<Dout>;
    fn offset<Din: DimTrait>(&self, stride: Din) -> usize;
}

#[cfg(test)]
mod shape_stride {
    use crate::dim;
    use crate::dim::default_stride;
    use crate::index::ShapeStride;

    #[test]
    fn is_transposed_false() {
        let shape = dim!(2, 3);
        let default_stride = default_stride(shape);

        let shape_stride = super::ShapeStride::new(shape, default_stride);

        assert!(!shape_stride.is_transposed());
    }

    #[test]
    fn is_transposed_true() {
        // transpose
        let shape_transposed = dim!(2, 3, 5, 4);
        let stride_transposed = dim!(60, 20, 1, 5);
        let shape_stride = ShapeStride::new(shape_transposed, stride_transposed);

        assert_eq!(shape_stride.is_transposed(), true);
    }
}

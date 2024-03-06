use crate::{
    dim::{cal_offset, DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::ToViewMutMemory,
    memory_impl::ViewMutMem,
    num::Num,
    shape_stride::ShapeStride,
};

struct MapAxis<'a, T: Num, F>
where
    F: FnMut(Matrix<ViewMutMem<T>, DimDyn>),
{
    matrix: Matrix<ViewMutMem<'a, T>, DimDyn>,
    axis: usize,
    fn_map: F,
}

impl<'a, T: Num, F> MapAxis<'a, T, F>
where
    F: FnMut(Matrix<ViewMutMem<T>, DimDyn>),
{
    fn new(matrix: Matrix<ViewMutMem<'a, T>, DimDyn>, axis: usize, fn_map: F) -> Self {
        Self {
            matrix,
            axis,
            fn_map,
        }
    }

    fn target_shape_stride(&self) -> ShapeStride<DimDyn> {
        let sh = self.target_shape();
        let st = self.target_stride();
        ShapeStride::new(DimDyn::from([sh]), DimDyn::from([st]))
    }

    fn target_stride(&self) -> usize {
        self.matrix.stride()[self.axis]
    }

    fn target_shape(&self) -> usize {
        self.matrix.shape()[self.axis]
    }

    /// fn_mapを適応する際に切り出すMatrixの一つ目の要素のIndexのVecを返す
    fn get_index(&self) -> Vec<DimDyn> {
        // axisのIndexを除いたIndexのVecを返す
        let mut candidates = Vec::with_capacity(self.matrix.shape().len() - 1);
        for (i, s) in self.matrix.shape().clone().into_iter().enumerate() {
            if i != self.axis {
                candidates.push(s);
            }
        }

        let combinations = generate_combinations(&candidates);
        combinations
            .into_iter()
            .map(|c| {
                let mut c = c;
                c.insert(self.axis, 0);
                DimDyn::from(c.as_slice())
            })
            .collect()
    }

    fn get_offsets(&self) -> Vec<usize> {
        self.get_index()
            .into_iter()
            .map(|itm| cal_offset(self.matrix.stride(), itm))
            .collect()
    }

    fn apply(&mut self) {
        let shapt_stride = self.target_shape_stride();
        for offset in self.get_offsets() {
            let m = self.matrix.memory_mut();
            let view = m.to_view_mut(offset);
            let matrix = Matrix::new(view, shapt_stride.shape(), shapt_stride.stride());
            (self.fn_map)(matrix);
        }
    }
}

fn generate_combinations(nums: &[usize]) -> Vec<Vec<usize>> {
    fn recurse(
        index: usize,
        current: &mut Vec<usize>,
        nums: &[usize],
        result: &mut Vec<Vec<usize>>,
    ) {
        if index == nums.len() {
            result.push(current.clone());
            return;
        }

        for i in 0..nums[index] {
            current.push(i);
            recurse(index + 1, current, nums, result);
            current.pop();
        }
    }

    let mut result = Vec::new();
    recurse(0, &mut Vec::new(), nums, &mut result);
    result
}

pub trait MatrixIter<T: Num> {
    fn map_axis<F>(&mut self, axis: usize, fn_map: F)
    where
        F: FnMut(Matrix<ViewMutMem<T>, DimDyn>);
}

impl<T: Num, M: ToViewMutMemory<Item = T>> MatrixIter<T> for Matrix<M, DimDyn> {
    fn map_axis<F>(&mut self, axis: usize, fn_map: F)
    where
        F: FnMut(Matrix<ViewMutMem<T>, DimDyn>),
    {
        if self.shape().len() <= 1 {
            panic!("shape.len() <= 1");
        }
        let mut_matrix = self.to_view_mut();
        let mut map_axis = MapAxis::new(mut_matrix, axis, fn_map);
        map_axis.apply();
    }
}

#[cfg(test)]
mod map_axis {
    use crate::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::{asum::Asum, copy_from::CopyFrom},
    };

    use super::MatrixIter;

    #[test]
    fn test_2d_0() {
        let mut a = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        a.map_axis(0, |m| {
            let mut m = m;
            let ans = OwnedMatrixDyn::from_vec(vec![2., 1.], [2]);
            CopyFrom::copy_from(&mut m, &ans.to_view());
        });

        let ans = OwnedMatrixDyn::from_vec(vec![2., 2., 2., 1., 1., 1.], [2, 3]);
        let diff = ans.to_view() - a.to_view();
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }

    #[test]
    fn test_2d_1() {
        let mut a = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        a.map_axis(1, |m| {
            let mut m = m;
            let ans = OwnedMatrixDyn::from_vec(vec![3., 2., 1.], [3]);
            CopyFrom::copy_from(&mut m, &ans.to_view());
        });

        let ans = OwnedMatrixDyn::from_vec(vec![3., 2., 1., 3., 2., 1.], [2, 3]);
        let diff = ans.to_view() - a.to_view();
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }

    #[test]
    fn test_3d_0() {
        let mut a = OwnedMatrixDyn::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [2, 2, 3],
        );

        a.map_axis(0, |m| {
            let mut m = m;
            let ans = OwnedMatrixDyn::from_vec(vec![2., 1.], [2]);
            CopyFrom::copy_from(&mut m, &ans.to_view());
        });

        println!("{:?}", a);

        let ans = OwnedMatrixDyn::from_vec(
            vec![2., 2., 2., 2., 2., 2., 1., 1., 1., 1., 1., 1.],
            [2, 2, 3],
        );

        let diff = ans.to_view() - a.to_view();
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }

    #[test]
    fn test_3d_1() {
        let mut a = OwnedMatrixDyn::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [2, 2, 3],
        );

        a.map_axis(1, |m| {
            let mut m = m;
            let ans = OwnedMatrixDyn::from_vec(vec![2., 1.], [2]);
            CopyFrom::copy_from(&mut m, &ans.to_view());
        });

        let ans = OwnedMatrixDyn::from_vec(
            vec![2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1.],
            [2, 2, 3],
        );

        let diff = ans.to_view() - a.to_view();
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }

    #[test]
    fn test_3d_2() {
        let mut a = OwnedMatrixDyn::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [2, 2, 3],
        );

        a.map_axis(2, |m| {
            let mut m = m;
            let ans = OwnedMatrixDyn::from_vec(vec![3., 2., 1.], [3]);
            CopyFrom::copy_from(&mut m, &ans.to_view());
        });

        let ans = OwnedMatrixDyn::from_vec(
            vec![3., 2., 1., 3., 2., 1., 3., 2., 1., 3., 2., 1.],
            [2, 2, 3],
        );

        let diff = ans.to_view() - a.to_view();
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }
}

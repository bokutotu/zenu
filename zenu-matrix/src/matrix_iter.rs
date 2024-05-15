use crate::{
    device::Device,
    dim::{cal_offset, DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref},
    num::Num,
    shape_stride::ShapeStride,
};

struct MapAxis<'a, T, F, D>
where
    T: Num,
    F: FnMut(Matrix<Ref<&'a mut T>, DimDyn, D>),
    D: Device,
{
    matrix: Matrix<Ref<&'a mut T>, DimDyn, D>,
    axis: usize,
    fn_map: F,
}

impl<'a, T, F, D> MapAxis<'a, T, F, D>
where
    T: Num,
    F: FnMut(Matrix<Ref<&'a mut T>, DimDyn, D>),
    D: Device,
{
    fn new(matrix: Matrix<Ref<&'a mut T>, DimDyn, D>, axis: usize, fn_map: F) -> Self {
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
        for (i, s) in self.matrix.shape().into_iter().enumerate() {
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
            // let m = self.matrix;
            let view = self.matrix.offset_ptr_mut(offset);
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

pub trait MatrixIter<T: Num, D: Device> {
    fn map_axis<F>(&self, axis: usize, fn_map: F) -> Matrix<Owned<T>, DimDyn, D>
    where
        F: FnMut(Matrix<Ref<&mut T>, DimDyn, D>);
    fn map_axis_mut<F>(self, axis: usize, fn_map: F)
    where
        F: FnMut(Matrix<Ref<&mut T>, DimDyn, D>);
}

impl<T: Num, D: Device> MatrixIter<T, D> for Matrix<Ref<&mut T>, DimDyn, D> {
    fn map_axis<F>(&self, axis: usize, fn_map: F) -> Matrix<Owned<T>, DimDyn, D>
    where
        F: FnMut(Matrix<Ref<&mut T>, DimDyn, D>),
    {
        let mut ans = Matrix::<_, DimDyn, D>::zeros(self.shape());
        ans.to_ref_mut().copy_from(&self);
        ans.to_ref_mut().map_axis_mut(axis, fn_map);
        ans
    }

    fn map_axis_mut<F>(self, axis: usize, fn_map: F)
    where
        F: FnMut(Matrix<Ref<&mut T>, DimDyn, D>),
    {
        if self.shape().len() <= 1 {
            panic!("shape.len() <= 1");
        }
        let mut map_axis = MapAxis::new(self, axis, fn_map);
        map_axis.apply();
    }
}

#[cfg(test)]
mod map_axis {
    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
        matrix_iter::MatrixIter,
    };

    fn test_2d_0<D: Device>() {
        let mut a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        a.to_ref_mut().map_axis_mut(0, |m| {
            let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![2., 1.], [2]);
            m.copy_from(&ans);
        });

        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![2., 2., 2., 1., 1., 1.], [2, 3]);
        let diff = ans - a;
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }
    #[test]
    fn test_2d_0_cpu() {
        test_2d_0::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn test_2d_0_cuda() {
        test_2d_0::<crate::device::nvidia::Nvidia>();
    }

    fn test_2d_1<D: Device>() {
        let mut a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        a.to_ref_mut().map_axis_mut(1, |m| {
            let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![3., 2., 1.], [3]);
            m.copy_from(&ans);
        });

        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![3., 2., 1., 3., 2., 1.], [2, 3]);
        let diff = ans - a;
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }
    #[test]
    fn test_2d_1_cpu() {
        test_2d_1::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn test_2d_1_cuda() {
        test_2d_1::<crate::device::nvidia::Nvidia>();
    }

    fn test_3d_0<D: Device>() {
        let mut a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [2, 2, 3],
        );

        a.to_ref_mut().map_axis_mut(0, |m| {
            let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![2., 1.], [2]);
            m.copy_from(&ans);
        });

        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![2., 2., 2., 2., 2., 2., 1., 1., 1., 1., 1., 1.],
            [2, 2, 3],
        );

        let diff = ans - a;
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }
    #[test]
    fn test_3d_0_cpu() {
        test_3d_0::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn test_3d_0_cuda() {
        test_3d_0::<crate::device::nvidia::Nvidia>();
    }

    fn test_3d_1<D: Device>() {
        let mut a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [2, 2, 3],
        );

        a.to_ref_mut().map_axis_mut(1, |m| {
            let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![2., 1.], [2]);
            m.copy_from(&ans);
        });

        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1.],
            [2, 2, 3],
        );

        let diff = ans - a;
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }
    #[test]
    fn test_3d_1_cpu() {
        test_3d_1::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn test_3d_1_cuda() {
        test_3d_1::<crate::device::nvidia::Nvidia>();
    }

    fn test_3d_2<D: Device>() {
        let mut a = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [2, 2, 3],
        );

        a.to_ref_mut().map_axis_mut(2, |m| {
            let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![3., 2., 1.], [3]);
            m.copy_from(&ans);
        });

        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![3., 2., 1., 3., 2., 1., 3., 2., 1., 3., 2., 1.],
            [2, 2, 3],
        );

        let diff = ans - a;
        let diff = diff.asum();
        assert_eq!(diff, 0.);
    }
    #[test]
    fn test_3d_2_cpu() {
        test_3d_2::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn test_3d_2_cuda() {
        test_3d_2::<crate::device::nvidia::Nvidia>();
    }
}

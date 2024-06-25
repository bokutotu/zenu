use crate::{
    device::Device,
    dim::{DimDyn, DimTrait, LessDimTrait},
    index::index_dyn_impl::Index,
    matrix::{Matrix, Owned, Ref},
    num::Num,
};

impl<T: Num, D: Device> Matrix<Ref<&T>, DimDyn, D> {
    pub fn sum(&self, axis: usize, keep_dim: bool) -> Matrix<Owned<T>, DimDyn, D> {
        let shape = self.shape();
        if axis >= shape.len() {
            panic!("Invalid axis");
        }

        let result_shape = self.shape().remove_axis(axis);

        let mut result = Matrix::zeros(result_shape);

        for i in 0..shape[axis] {
            let mut result_view_mut = result.to_ref_mut();
            let s = self.clone();
            let s = s.index_axis_dyn(Index::new(axis, i));
            result_view_mut.add_assign(&s);
        }

        if keep_dim {
            result.add_axis(axis);
        }
        result
    }
}

pub fn sum_to<T: Num, D: Device>(
    source: Matrix<Ref<&T>, DimDyn, D>,
    target: Matrix<Ref<&mut T>, DimDyn, D>,
) {
    if source.shape().len() < target.shape().len() {
        panic!("source.shape().len() < target.shape().len()");
    }

    let diff_len = source.shape().len() - target.shape().len();
    if diff_len == 0 {
        let target = target;
        target.copy_from(&source);
        return;
    }

    if !source.shape().is_include(target.shape()) {
        panic!("!source.shape().is_include(target.shape())");
    }

    if diff_len == 1 {
        let target = target;
        let ans = source.sum(0, false);
        target.copy_from(&ans);
    } else {
        sum_to(source.sum(0, false).to_ref(), target);
    }
}

#[cfg(test)]
mod sum {
    use crate::{
        device::Device,
        dim::{DimDyn, DimTrait},
        matrix::{Matrix, Owned},
    };

    fn test_4d<D: Device>() {
        let mut source_vec = Vec::new();
        for i in 0..2 * 3 * 4 * 5 {
            source_vec.push(i as f32);
        }
        let source: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(source_vec, [2, 3, 4, 5]);

        let sum_0 = source.to_ref().sum(0, false);
        let sum_1 = source.to_ref().sum(1, false);
        let sum_2 = source.to_ref().sum(2, false);
        let sum_3 = source.to_ref().sum(3, false);

        assert_eq!(sum_0.shape().slice(), [3, 4, 5]);
        assert_eq!(sum_1.shape().slice(), [2, 4, 5]);
        assert_eq!(sum_2.shape().slice(), [2, 3, 5]);
        assert_eq!(sum_3.shape().slice(), [2, 3, 4]);

        let mut ans_vec_0 = Vec::new();
        for i in 60..=178 {
            if i % 2 == 0 {
                ans_vec_0.push(i as f32);
            }
        }
        let ans_0: Matrix<_, DimDyn, _> = Matrix::from_vec(ans_vec_0, [3, 4, 5]);
        let diff = sum_0.to_ref() - ans_0.to_ref();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);

        let ans_vec_1 = vec![
            60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117,
            240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270, 273, 276, 279, 282, 285, 288,
            291, 294, 297,
        ];
        let nas_vec_1 = ans_vec_1.into_iter().map(|x| x as f32).collect();
        let ans_1: Matrix<_, DimDyn, _> = Matrix::from_vec(nas_vec_1, [2, 4, 5]);
        let diff = sum_1.to_ref() - ans_1.to_ref();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);

        let ans_vec_2 = vec![
            30, 34, 38, 42, 46, 110, 114, 118, 122, 126, 190, 194, 198, 202, 206, 270, 274, 278,
            282, 286, 350, 354, 358, 362, 366, 430, 434, 438, 442, 446,
        ];
        let nas_vec_2 = ans_vec_2.into_iter().map(|x| x as f32).collect();
        let ans_2 = Matrix::<_, DimDyn, _>::from_vec(nas_vec_2, [2, 3, 5]);
        let diff = sum_2.to_ref() - ans_2.to_ref();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);

        let ans_vec_3 = vec![
            10, 35, 60, 85, 110, 135, 160, 185, 210, 235, 260, 285, 310, 335, 360, 385, 410, 435,
            460, 485, 510, 535, 560, 585,
        ];
        let nas_vec_3 = ans_vec_3.into_iter().map(|x| x as f32).collect();
        let ans_3 = Matrix::<_, DimDyn, _>::from_vec(nas_vec_3, [2, 3, 4]);
        let diff = sum_3.to_ref() - ans_3.to_ref();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);
    }
    #[test]
    fn test_4d_cpu() {
        test_4d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn test_4d_gpu() {
        test_4d::<crate::device::nvidia::Nvidia>();
    }

    fn test_4d_keep_dim<D: Device>() {
        let mut source_vec = Vec::new();
        for i in 0..2 * 3 * 4 * 5 {
            source_vec.push(i as f32);
        }
        let source = Matrix::<_, DimDyn, D>::from_vec(source_vec, [2, 3, 4, 5]);

        let sum_0 = source.to_ref().sum(0, true);
        let sum_1 = source.to_ref().sum(1, true);
        let sum_2 = source.to_ref().sum(2, true);
        let sum_3 = source.to_ref().sum(3, true);

        assert_eq!(sum_0.shape().slice(), [1, 3, 4, 5]);
        assert_eq!(sum_1.shape().slice(), [2, 1, 4, 5]);
        assert_eq!(sum_2.shape().slice(), [2, 3, 1, 5]);
        assert_eq!(sum_3.shape().slice(), [2, 3, 4, 1]);

        let mut ans_vec_0 = Vec::new();
        for i in 60..=178 {
            if i % 2 == 0 {
                ans_vec_0.push(i as f32);
            }
        }
        let ans_0 = Matrix::<_, DimDyn, D>::from_vec(ans_vec_0, [1, 3, 4, 5]);
        let diff = sum_0.to_ref() - ans_0.to_ref();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);

        let ans_vec_1 = vec![
            60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117,
            240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270, 273, 276, 279, 282, 285, 288,
            291, 294, 297,
        ];
        let nas_vec_1 = ans_vec_1.into_iter().map(|x| x as f32).collect();
        let ans_1 = Matrix::<_, DimDyn, D>::from_vec(nas_vec_1, [2, 1, 4, 5]);
        let diff = sum_1.to_ref() - ans_1.to_ref();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);

        let ans_vec_2 = vec![
            30, 34, 38, 42, 46, 110, 114, 118, 122, 126, 190, 194, 198, 202, 206, 270, 274, 278,
            282, 286, 350, 354, 358, 362, 366, 430, 434, 438, 442, 446,
        ];
        let nas_vec_2 = ans_vec_2.into_iter().map(|x| x as f32).collect();
        let ans_2 = Matrix::<_, DimDyn, D>::from_vec(nas_vec_2, [2, 3, 1, 5]);
        let diff = sum_2.to_ref() - ans_2.to_ref();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);

        let ans_vec_3 = vec![
            10, 35, 60, 85, 110, 135, 160, 185, 210, 235, 260, 285, 310, 335, 360, 385, 410, 435,
            460, 485, 510, 535, 560, 585,
        ];
        let nas_vec_3 = ans_vec_3.into_iter().map(|x| x as f32).collect();
        let ans_3 = Matrix::<_, DimDyn, D>::from_vec(nas_vec_3, [2, 3, 4, 1]);
        let diff = sum_3.to_ref() - ans_3.to_ref();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);
    }
    #[test]
    fn test_4d_keep_dim_cpu() {
        test_4d_keep_dim::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn test_4d_keep_dim_gpu() {
        test_4d_keep_dim::<crate::device::nvidia::Nvidia>();
    }
}

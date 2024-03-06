use crate::{
    dim::{DimDyn, DimTrait},
    index::index_dyn_impl::Index,
    matrix::{IndexAxisDyn, MatrixBase, OwnedMatrix, ToViewMatrix, ToViewMutMatrix, ViewMatrix},
    matrix_impl::Matrix,
    memory_impl::{OwnedMem, ViewMem},
    num::Num,
    operation::zeros::Zeros,
};

use super::{add::MatrixAdd, copy_from::CopyFrom};

pub trait MatrixSum: ViewMatrix {
    type Output: OwnedMatrix;
    fn sum(self, axis: usize) -> Self::Output;
}

impl<'a, T: Num, D: DimTrait> MatrixSum for Matrix<ViewMem<'a, T>, D> {
    type Output = Matrix<OwnedMem<T>, DimDyn>;
    fn sum(self, axis: usize) -> Self::Output {
        let self_dyn = self.into_dyn_dim();
        let shape = self_dyn.shape();
        if axis >= shape.len() {
            panic!("Invalid axis");
        }
        let result_shape = {
            let mut shape_ = DimDyn::default();
            for (i, &s) in shape.slice().iter().enumerate() {
                if i != axis {
                    shape_.push_dim(s);
                }
            }
            shape_
        };

        let mut result = Self::Output::zeros(result_shape);

        for i in 0..shape[axis] {
            let result_view_mut = result.to_view_mut();
            let s = self_dyn.clone();
            let s = s.index_axis_dyn(Index::new(axis, i));
            let tmp = result_view_mut.add(s);
            result.to_view_mut().copy_from(&tmp.to_view());
        }

        result
    }
}

#[cfg(test)]
mod sum {
    use crate::{
        dim::DimTrait,
        matrix::{MatrixBase, OwnedMatrix, ToViewMatrix},
        matrix_impl::{OwnedMatrix3D, OwnedMatrix4D},
        operation::{asum::Asum, sum::MatrixSum},
    };

    #[test]
    fn test_4d() {
        let mut source_vec = Vec::new();
        for i in 0..2 * 3 * 4 * 5 {
            source_vec.push(i as f32);
        }
        let source = OwnedMatrix4D::from_vec(source_vec, [2, 3, 4, 5]);

        let sum_0 = source.clone().to_view().sum(0);
        let sum_1 = source.clone().to_view().sum(1);
        let sum_2 = source.clone().to_view().sum(2);
        let sum_3 = source.clone().to_view().sum(3);

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
        let ans_0 = OwnedMatrix3D::from_vec(ans_vec_0, [3, 4, 5]);
        println!("{:?}", sum_0.to_view());
        println!("{:?}", ans_0.to_view());
        let diff = sum_0.to_view() - ans_0.to_view();
        let diff_sum = Asum::asum(diff);
        assert!(diff_sum < 1e-6);

        let ans_vec_1 = vec![
            60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117,
            240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270, 273, 276, 279, 282, 285, 288,
            291, 294, 297,
        ];
        let nas_vec_1 = ans_vec_1.into_iter().map(|x| x as f32).collect();
        let ans_1 = OwnedMatrix3D::from_vec(nas_vec_1, [2, 4, 5]);
        let diff = sum_1.to_view() - ans_1.to_view();
        let diff_sum = Asum::asum(diff);
        assert!(diff_sum < 1e-6);

        let ans_vec_2 = vec![
            30, 34, 38, 42, 46, 110, 114, 118, 122, 126, 190, 194, 198, 202, 206, 270, 274, 278,
            282, 286, 350, 354, 358, 362, 366, 430, 434, 438, 442, 446,
        ];
        let nas_vec_2 = ans_vec_2.into_iter().map(|x| x as f32).collect();
        let ans_2 = OwnedMatrix3D::from_vec(nas_vec_2, [2, 3, 5]);
        let diff = sum_2.to_view() - ans_2.to_view();
        let diff_sum = Asum::asum(diff);
        assert!(diff_sum < 1e-6);

        let ans_vec_3 = vec![
            10, 35, 60, 85, 110, 135, 160, 185, 210, 235, 260, 285, 310, 335, 360, 385, 410, 435,
            460, 485, 510, 535, 560, 585,
        ];
        let nas_vec_3 = ans_vec_3.into_iter().map(|x| x as f32).collect();
        let ans_3 = OwnedMatrix3D::from_vec(nas_vec_3, [2, 3, 4]);
        let diff = sum_3.to_view() - ans_3.to_view();
        let diff_sum = Asum::asum(diff);
        assert!(diff_sum < 1e-6);
    }
}

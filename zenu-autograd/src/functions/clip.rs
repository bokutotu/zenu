use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    num::Num,
    operation::{
        clip::{clip_filter, Clip as C},
        copy_from::CopyFrom,
    },
};

use crate::{Function, Variable, VariableWeak};

struct Clip<T: Num> {
    min: T,
    max: T,
    input: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> Clip<T> {
    pub fn new(min: T, max: T, input: Variable<T>, output: Variable<T>) -> Self {
        assert_eq!(
            input.get_data().shape(),
            output.get_data().shape(),
            "input.shape() != output.shape()"
        );
        let output = output.downgrade();
        Self {
            max,
            min,
            input,
            output,
        }
    }
}

impl<T: Num> Function<T> for Clip<T> {
    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().clone().unwrap();
        let clip_filter = clip_filter(self.input.get_data(), self.min, self.max);
        let clip_filter = Variable::from(clip_filter);
        let input_grad = output_grad * clip_filter;
        self.input.set_grad(input_grad);
    }

    fn forward(&self) {
        let input = self.input.get_data();
        let output = C::clip(&input, self.min, self.max);
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_view_mut()
            .copy_from(&output.to_view());
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.input.clone()]
    }
}

pub fn clip<T: Num>(input: Variable<T>, min: T, max: T) -> Variable<T> {
    let output = Variable::new(input.get_data().clone());
    let clip = Clip::new(min, max, input, output.clone());
    clip.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(clip))));
    output
}

#[cfg(test)]
mod clip {
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use crate::creator::from_vec::from_vec;

    #[test]
    fn clip_1d() {
        let input = from_vec(vec![1., 2., 3., 4., 5., 6.], [6]);
        let output = super::clip(input.clone(), 2.0, 4.0);
        output.backward();
        let output = dbg!(output.get_data());
        let ans = OwnedMatrixDyn::from_vec(vec![2., 2., 3., 4., 4., 4.], [6]);
        let diff = output - ans;
        let diff_asum = diff.asum();
        assert_eq!(diff_asum, 0.0);
        let input_grad = input.get_grad().unwrap().get_data();
        let ans = OwnedMatrixDyn::from_vec(vec![0., 1., 1., 1., 0., 0.], [6]);
        let diff = input_grad - ans;
        let diff_asum = diff.asum();
        assert_eq!(diff_asum, 0.0);
    }
}

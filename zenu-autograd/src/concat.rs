use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    concat::concat as c,
    device::Device,
    dim::{DimDyn, DimTrait},
    index::index_dyn_impl::Index,
    num::Num,
};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

struct Concat<T: Num, D: Device> {
    vars: Vec<Variable<T, D>>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for Concat<T, D> {
    fn forward(&self) {
        let matrix = self
            .vars
            .iter()
            .map(|v| v.get_data().clone())
            .collect::<Vec<_>>();
        let matrix = c(&matrix);
        self.output.upgrade().unwrap().swap_inner(matrix);
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().unwrap().clone();
        let input_grads = concat_grad(output_grad);
        self.vars
            .iter()
            .zip(input_grads)
            .for_each(|(v, g)| v.set_grad(g));
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        self.vars.clone()
    }
}

struct ConcatGrad<T: Num, D: Device> {
    input: Variable<T, D>,
    output: Vec<VariableWeak<T, D>>,
}

impl<T: Num, D: Device> Function<T, D> for ConcatGrad<T, D> {
    fn forward(&self) {
        for (i, output) in self.output.iter().enumerate() {
            let input_mat = self.input.get_as_ref();
            let input_slice = input_mat.index_axis(Index::new(0, i));

            output
                .upgrade()
                .unwrap()
                .get_as_mut()
                .copy_from(&input_slice);
        }
    }

    fn backward(&self) {
        let output_grads = self
            .output
            .iter()
            .map(|v| v.upgrade().unwrap().get_grad().unwrap().clone())
            .collect::<Vec<_>>();

        let input_grad = concat(&output_grads);
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

#[must_use]
pub fn concat<T: Num, D: Device>(vars: &[Variable<T, D>]) -> Variable<T, D> {
    let output = Variable::new(c(&vars
        .iter()
        .map(|v| v.get_data().clone())
        .collect::<Vec<_>>()));

    let concat = Concat {
        vars: vars.to_vec(),
        output: output.clone().downgrade(),
    };

    output.set_creator(Rc::new(RefCell::new(Box::new(concat))));
    output
}

fn concat_grad<T: Num, D: Device>(input: Variable<T, D>) -> Vec<Variable<T, D>> {
    let input_shape = input.get_shape();
    let output_shape_slice = &input_shape.slice()[1..];
    let output_shape = DimDyn::from(output_shape_slice);

    let mut output = Vec::with_capacity(output_shape[0]);
    for _ in 0..output_shape[0] {
        output.push(zeros(output_shape));
    }

    let concat_grad = ConcatGrad {
        input,
        output: output.iter().map(|v| v.clone().downgrade()).collect(),
    };

    let layer = Rc::new(RefCell::new(
        Box::new(concat_grad) as Box<dyn Function<T, D>>
    ));
    output.iter().for_each(|v| v.set_creator(layer.clone()));
    output
}

#[cfg(test)]
mod concat_test {
    use zenu_matrix::{
        device::cpu::Cpu,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::assert_val_eq;

    use crate::Variable;

    #[expect(clippy::many_single_char_names)]
    #[test]
    fn fwd() {
        let x_mat = Matrix::<Owned<f32>, _, Cpu>::ones([2]);
        let y_mat = Matrix::zeros([2]);
        let t_mat = Matrix::from_vec(vec![1., 2., 3., 4.], [2, 2]);

        let x = Variable::new(x_mat);
        let y = Variable::new(y_mat);
        let t = Variable::new(t_mat);

        let z = super::concat(&[x.clone(), y.clone()]);
        let w = z.clone() * t;
        w.backward();

        let z_expected = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(vec![1., 1., 0., 0.], [2, 2]);

        assert_val_eq!(z, z_expected, 1e-5);
    }
}

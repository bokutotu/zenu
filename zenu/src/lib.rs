pub mod dataset;
pub mod dataset_loader;

use std::path::Path;

use zenu_autograd::{creator::zeros::zeros, Variable};
use zenu_matrix::{
    device::Device,
    dim::DimDyn,
    matrix::{Matrix, Owned},
    num::Num,
};
use zenu_optimizer::Optimizer;

pub trait Model<T: Num, D: Device> {
    fn predict(&self, inputs: &[Variable<T, D>]) -> Variable<T, D>;
}

pub fn update_parameters<T: Num, D: Device, O: Optimizer<T, D>>(
    loss: Variable<T, D>,
    optimizer: &O,
) {
    loss.backward();
    let parameters = loss.get_all_trainable_variables();
    optimizer.update(&parameters);
    loss.clear_grad();
}

pub fn save_model<T: Num, D: Device, M: Model<T, D>, P: AsRef<Path>>(
    model: M,
    input_shape: &[usize],
    save_path: P,
) -> Result<(), &'static str> {
    let zeros = zeros(input_shape);
    let output = model.predict(&[zeros]);
    let parameters = output.get_all_trainable_variables();
    let parameters_mat = parameters
        .into_iter()
        .map(|x| x.get_data().clone())
        .collect::<Vec<_>>();
    Ok(())
}

pub fn load_model<T, D: Device, M: Model<T, D>, P: AsRef<Path>>(
    load_path: P,
    model: &mut M,
    input_shape: &[usize],
) -> Result<(), &'static str>
where
    T: serde::de::DeserializeOwned + Num,
    M: Model<T, D>,
    P: AsRef<Path>,
{
    let json = std::fs::read_to_string(load_path).map_err(|_| "Failed to load model")?;
    // let parameters_mat: Vec<Matrix<Owned<T>, DimDyn, D>> = serde_json::from_str(&json).unwrap();
    // let zeros = zeros(input_shape);
    // let output = model.predict(&[zeros]);
    // let parameters = output.get_all_trainable_variables();
    // parameters
    //     .into_iter()
    //     .zip(parameters_mat)
    //     .for_each(|(x, y)| x.get_data_mut().copy_from(&y));
    Ok(())
}

#[cfg(test)]
mod save_and_load_paramters {
    use super::{load_model, save_model, Model};
    use zenu_autograd::creator::rand;
    use zenu_layer::{layers::linear::Linear, Module};
    use zenu_matrix::device::Device;

    // #[test]
    fn save_and_load_parameters<D: Device>() {
        struct TestModel<D: Device> {
            layer1: Linear<f32, D>,
            layer2: Linear<f32, D>,
        }

        impl<D: Device> Model<f32, D> for TestModel<D> {
            fn predict(
                &self,
                inputs: &[zenu_autograd::Variable<f32, D>],
            ) -> zenu_autograd::Variable<f32, D> {
                let x = self.layer1.call(inputs[0].clone());
                self.layer2.call(x)
            }
        }

        let mut model = TestModel::<D> {
            layer1: Linear::new(2, 2, true),
            layer2: Linear::new(2, 2, true),
        };

        let input_shape = [1, 2];
        let save_path = "test.json";
        save_model(model, &input_shape, save_path).unwrap();

        let mut model = TestModel::<D> {
            layer1: Linear::new(2, 2, true),
            layer2: Linear::new(2, 2, true),
        };
        load_model(save_path, &mut model, &input_shape).unwrap();

        let fake_input = rand::normal(1.0, 1.0, Some(42), &[1, 2]);
        let original = model.predict(&[fake_input.clone()]);
        let loaded = model.predict(&[fake_input]);

        // assert!(
        //     (original.get_data() - loaded.get_data()).asum() < 1e-6,
        //     "Failed to load parameters"
        // );

        std::fs::remove_file(save_path).unwrap();
    }
}

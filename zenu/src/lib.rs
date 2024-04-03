pub mod dataset;
pub mod dataset_loader;

use std::path::Path;

use zenu_autograd::{creator::zeros::zeros, Variable};
use zenu_matrix::{matrix_impl::OwnedMatrixDyn, num::Num, operation::copy_from::CopyFrom};
use zenu_optimizer::Optimizer;

pub trait Model<T: Num> {
    fn predict(&self, inputs: &[Variable<T>]) -> Variable<T>;
}

pub fn update_parameters<T: Num, O: Optimizer<T>>(loss: Variable<T>, optimizer: &O) {
    loss.backward();
    let parameters = loss.get_all_trainable_variables();
    optimizer.update(&parameters);
    loss.clear_grad();
}

pub fn save_model<T: Num, M: Model<T>, P: AsRef<Path>>(
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
    let save_json = serde_json::to_string(&parameters_mat).unwrap();
    std::fs::write(save_path, save_json).map_err(|_| "Failed to save model")?;
    Ok(())
}

pub fn load_model<T, M: Model<T>, P: AsRef<Path>>(
    load_path: P,
    model: &mut M,
    input_shape: &[usize],
) -> Result<(), &'static str>
where
    T: serde::de::DeserializeOwned + Num,
    M: Model<T>,
    P: AsRef<Path>,
{
    let json = std::fs::read_to_string(load_path).map_err(|_| "Failed to load model")?;
    let parameters_mat: Vec<OwnedMatrixDyn<T>> = serde_json::from_str(&json).unwrap();
    let zeros = zeros(input_shape);
    let output = model.predict(&[zeros]);
    let parameters = output.get_all_trainable_variables();
    parameters
        .into_iter()
        .zip(parameters_mat)
        .for_each(|(x, y)| x.get_data_mut().copy_from(&y));
    Ok(())
}

#[cfg(test)]
mod save_and_load_paramters {
    use super::{load_model, save_model, Model};
    use zenu_autograd::creator::rand;
    use zenu_layer::{layers::linear::Linear, Layer};
    use zenu_matrix::operation::asum::Asum;

    #[test]
    fn save_and_load_parameters() {
        struct TestModel {
            layer1: Linear<f32>,
            layer2: Linear<f32>,
        }

        impl Model<f32> for TestModel {
            fn predict(
                &self,
                inputs: &[zenu_autograd::Variable<f32>],
            ) -> zenu_autograd::Variable<f32> {
                let x = self.layer1.call(inputs[0].clone());
                self.layer2.call(x)
            }
        }

        let mut model = TestModel {
            layer1: Linear::new(2, 2),
            layer2: Linear::new(2, 2),
        };

        model.layer1.init_parameters(Some(42));
        model.layer2.init_parameters(Some(42));
        let input_shape = [1, 2];
        let save_path = "test.json";
        save_model(model, &input_shape, save_path).unwrap();

        let mut model = TestModel {
            layer1: Linear::new(2, 2),
            layer2: Linear::new(2, 2),
        };
        model.layer1.init_parameters(Some(42));
        model.layer2.init_parameters(Some(42));
        load_model(save_path, &mut model, &input_shape).unwrap();

        let fake_input = rand::normal(1.0, 1.0, Some(42), &[1, 2]);
        let original = model.predict(&[fake_input.clone()]);
        let loaded = model.predict(&[fake_input]);

        assert!(
            (original.get_data() - loaded.get_data()).asum() < 1e-6,
            "Failed to load parameters"
        );

        std::fs::remove_file(save_path).unwrap();
    }
}

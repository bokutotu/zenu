pub mod dataset;
pub mod dataset_loader;

use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
};

use serde::Deserialize;

use zenu_autograd::Variable;
use zenu_layer::Parameters;
use zenu_matrix::{device::Device, num::Num};

pub extern crate zenu_macros;

pub use zenu_autograd as autograd;
pub use zenu_layer as layer;
pub use zenu_macros as macros;
pub use zenu_matrix as matrix;
pub use zenu_optimizer as optimizer;

#[expect(clippy::missing_errors_doc)]
pub fn save_model<T: Num, D: Device, M: Parameters<T, D>, P: AsRef<Path>>(
    model: &M,
    path: P,
) -> Result<(), &'static str> {
    let path = path.as_ref();
    let mut file = File::create(path).map_err(|_| "Failed to save model")?;
    let state_dict = model.parameters();
    let bin = bincode::serialize(&state_dict).map_err(|_| "Failed to save model")?;
    file.write_all(&bin).map_err(|_| "Failed to save model")?;
    Ok(())
}

#[expect(clippy::missing_errors_doc)]
pub fn load_model<
    T: Num + for<'de> Deserialize<'de>,
    D: Device,
    M: Parameters<T, D>,
    P: AsRef<Path>,
>(
    path: P,
    model: &M,
) -> Result<(), &'static str> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|_| "Failed to load model")?;
    let mut bin = Vec::new();
    file.read_to_end(&mut bin)
        .map_err(|_| "Failed to load model")?;
    let state_dict: HashMap<String, Variable<T, D>> =
        bincode::deserialize(&bin).map_err(|_| "Failed to load model")?;
    let model_state_dict = model.parameters();
    for (key, value) in &state_dict {
        if let Some(parameter) = model_state_dict.get(key) {
            parameter
                .get_data_mut()
                .to_ref_mut()
                .copy_from(&value.get_data());
        } else {
            return Err("Failed to load model");
        }
    }
    Ok(())
}

#[cfg(test)]
mod save_and_load_paramters {
    use std::collections::HashMap;

    use super::{load_model, save_model};
    use zenu_autograd::{creator::rand, Variable};
    use zenu_layer::{layers::linear::Linear, Module, Parameters};
    use zenu_matrix::{
        device::{cpu::Cpu, Device},
        num::Num,
    };
    use zenu_test::run_test;

    #[test]
    fn save_and_load_parameters() {
        struct TestModel {
            layer1: Linear<f32, Cpu>,
            layer2: Linear<f32, Cpu>,
        }

        impl Module<f32, Cpu> for TestModel {
            type Input = Variable<f32, Cpu>;
            type Output = Variable<f32, Cpu>;
            fn call(&self, inputs: Variable<f32, Cpu>) -> zenu_autograd::Variable<f32, Cpu> {
                let x = self.layer1.call(inputs.clone());
                self.layer2.call(x)
            }
        }

        impl Parameters<f32, Cpu> for TestModel {
            fn weights(&self) -> HashMap<String, Variable<f32, Cpu>> {
                let mut weights = HashMap::new();
                let layer1_weight = self.layer1.weight.clone();
                let layer2_weight = self.layer2.weight.clone();
                weights.insert("layer1.weight".to_string(), layer1_weight);
                weights.insert("layer2.weight".to_string(), layer2_weight);
                weights
            }
            fn biases(&self) -> HashMap<String, Variable<f32, Cpu>> {
                let mut biases = HashMap::new();
                biases.insert("layer1.bias".to_string(), self.layer1.bias.clone().unwrap());
                biases.insert("layer2.bias".to_string(), self.layer2.bias.clone().unwrap());
                biases
            }
        }

        let model = TestModel {
            layer1: Linear::new(2, 2, true),
            layer2: Linear::new(2, 2, true),
        };

        let save_path = "test.json";
        save_model(&model, save_path).unwrap();

        let new_model = TestModel {
            layer1: Linear::new(2, 2, true),
            layer2: Linear::new(2, 2, true),
        };
        load_model(save_path, &new_model).unwrap();

        let fake_input = rand::normal(1.0, 1.0, Some(42), [1, 2]);
        let original = model.call(fake_input.clone());
        let loaded = model.call(fake_input);

        assert!(
            (original.get_data().to_ref() - loaded.get_data().to_ref()).asum() < 1e-6,
            "Failed to load parameters"
        );

        std::fs::remove_file(save_path).unwrap();
    }

    struct ConvNet<T: Num, D: Device> {
        conv1: zenu_layer::layers::conv2d::Conv2d<T, D>,
        conv2: zenu_layer::layers::conv2d::Conv2d<T, D>,
        fc1: Linear<T, D>,
        fc2: Linear<T, D>,
    }

    impl<T: Num, D: Device> Parameters<T, D> for ConvNet<T, D> {
        fn weights(&self) -> HashMap<String, Variable<T, D>> {
            let mut weights = HashMap::new();
            let conv1_weights = self.conv1.weights();
            let conv2_weights = self.conv2.weights();
            let fc1_weights = self.fc1.weights();
            let fc2_weights = self.fc2.weights();

            weights.insert(
                "conv1.filter".to_string(),
                conv1_weights["conv2d.filter"].clone(),
            );
            weights.insert(
                "conv2.filter".to_string(),
                conv2_weights["conv2d.filter"].clone(),
            );
            weights.insert(
                "fc1.weight".to_string(),
                fc1_weights["linear.weight"].clone(),
            );
            weights.insert(
                "fc2.weight".to_string(),
                fc2_weights["linear.weight"].clone(),
            );
            weights
        }

        fn biases(&self) -> HashMap<String, Variable<T, D>> {
            let mut biases = HashMap::new();
            let conv1_biases = self.conv1.biases();
            let conv2_biases = self.conv2.biases();
            let fc1_biases = self.fc1.biases();
            let fc2_biases = self.fc2.biases();

            biases.insert(
                "conv1.bias".to_string(),
                conv1_biases["conv2d.bias"].clone(),
            );
            biases.insert(
                "conv2.bias".to_string(),
                conv2_biases["conv2d.bias"].clone(),
            );
            biases.insert("fc1.bias".to_string(), fc1_biases["linear.bias"].clone());
            biases.insert("fc2.bias".to_string(), fc2_biases["linear.bias"].clone());
            biases
        }
    }

    fn hoge<D: Device>() {
        let model = ConvNet::<f32, D> {
            conv1: zenu_layer::layers::conv2d::Conv2d::new(
                1,
                20,
                (5, 5),
                (1, 1),
                (0, 0),
                (1, 1),
                true,
            ),
            conv2: zenu_layer::layers::conv2d::Conv2d::new(
                20,
                50,
                (5, 5),
                (1, 1),
                (0, 0),
                (1, 1),
                true,
            ),
            fc1: Linear::new(4 * 4 * 50, 500, true),
            fc2: Linear::new(500, 10, true),
        };

        let save_path = "test.json";
        save_model(&model, save_path).unwrap();

        let model = ConvNet::<f32, D> {
            conv1: zenu_layer::layers::conv2d::Conv2d::new(
                1,
                20,
                (5, 5),
                (1, 1),
                (0, 0),
                (1, 1),
                true,
            ),
            conv2: zenu_layer::layers::conv2d::Conv2d::new(
                20,
                50,
                (5, 5),
                (1, 1),
                (0, 0),
                (1, 1),
                true,
            ),
            fc1: Linear::new(4 * 4 * 50, 500, true),
            fc2: Linear::new(500, 10, true),
        };

        load_model::<f32, D, ConvNet<f32, D>, _>(save_path, &model).unwrap();

        std::fs::remove_file(save_path).unwrap();
    }
    run_test!(hoge, hoge_cpu, hoge_nvidia);
}

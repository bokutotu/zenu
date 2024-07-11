pub mod dataset;
pub mod dataset_loader;

use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use serde::Deserialize;
use zenu_autograd::Variable;
use zenu_layer::StateDict;
use zenu_matrix::{device::Device, num::Num};
use zenu_optimizer::Optimizer;

pub use zenu_autograd as autograd;
pub use zenu_layer as layer;
pub use zenu_matrix as matrix;
pub use zenu_optimizer as optimizer;

pub fn update_parameters<T: Num, D: Device, O: Optimizer<T, D>>(
    loss: Variable<T, D>,
    optimizer: &O,
) {
    loss.backward();
    let parameters = loss.get_all_trainable_variables();
    optimizer.update(&parameters);
    loss.clear_grad();
}

pub fn save_model<'de, M: StateDict<'de>, P: AsRef<Path>>(
    model: M,
    save_path: P,
) -> Result<(), &'static str> {
    let bin = model.to_bytes();
    let mut file = File::create(save_path).map_err(|_| "Failed to save model")?;
    file.write_all(&bin).map_err(|_| "Failed to save model")?;
    Ok(())
}

pub fn load_model_from_vec<'de, T, D: Device, M: StateDict<'de>>(
    bin: &'de [u8],
) -> Result<M, &'static str>
where
    T: Num + Deserialize<'de>,
{
    let model = M::from_bytes(bin);
    Ok(model)
}

pub fn load_model<M: for<'de> StateDict<'de>, P: AsRef<Path>>(
    save_path: P,
) -> Result<M, &'static str> {
    let mut file = File::open(save_path).map_err(|_| "Failed to load model")?;
    let mut bin = Vec::new();
    file.read_to_end(&mut bin)
        .map_err(|_| "Failed to load model")?;

    Ok(M::from_bytes(&bin))
}

#[cfg(test)]
mod save_and_load_paramters {
    use super::{load_model, save_model};
    use serde::{Deserialize, Serialize};
    use zenu_autograd::{creator::rand, Variable};
    use zenu_layer::{layers::linear::Linear, Module, StateDict};
    use zenu_matrix::{
        device::{cpu::Cpu, Device},
        num::Num,
    };
    use zenu_test::run_test;

    #[test]
    fn save_and_load_parameters() {
        #[derive(Serialize, Deserialize)]
        struct TestModel {
            layer1: Linear<f32, Cpu>,
            layer2: Linear<f32, Cpu>,
        }

        impl<'de> StateDict<'de> for TestModel {}

        impl Module<f32, Cpu> for TestModel {
            fn call(&self, inputs: Variable<f32, Cpu>) -> zenu_autograd::Variable<f32, Cpu> {
                let x = self.layer1.call(inputs.clone());
                self.layer2.call(x)
            }
        }

        let model = TestModel {
            layer1: Linear::new(2, 2, true),
            layer2: Linear::new(2, 2, true),
        };

        let save_path = "test.json";
        save_model(model, save_path).unwrap();

        let model = load_model::<TestModel, _>(save_path).unwrap();

        let fake_input = rand::normal(1.0, 1.0, Some(42), &[1, 2]);
        let original = model.call(fake_input.clone());
        let loaded = model.call(fake_input);

        assert!(
            (original.get_data().to_ref() - loaded.get_data().to_ref()).asum() < 1e-6,
            "Failed to load parameters"
        );

        std::fs::remove_file(save_path).unwrap();
    }

    #[derive(Serialize, Deserialize)]
    #[serde(bound(deserialize = "T: Num + Deserialize<'de>, D: Device + Deserialize<'de>"))]
    struct ConvNet<T: Num, D: Device> {
        conv1: zenu_layer::layers::conv2d::Conv2d<T, D>,
        conv2: zenu_layer::layers::conv2d::Conv2d<T, D>,
        fc1: Linear<T, D>,
        fc2: Linear<T, D>,
    }

    impl<'de, T: Num + Deserialize<'de>, D: Device + Deserialize<'de>> StateDict<'de>
        for ConvNet<T, D>
    {
    }

    fn hoge<D: Device + for<'de> Deserialize<'de>>() {
        let model = ConvNet::<f32, D> {
            conv1: zenu_layer::layers::conv2d::Conv2d::new(1, 20, (5, 5), (1, 1), (0, 0), true),
            conv2: zenu_layer::layers::conv2d::Conv2d::new(20, 50, (5, 5), (1, 1), (0, 0), true),
            fc1: Linear::new(4 * 4 * 50, 500, true),
            fc2: Linear::new(500, 10, true),
        };

        let save_path = "test.json";
        save_model(model, save_path).unwrap();

        let _model = load_model::<ConvNet<f32, D>, _>(save_path).unwrap();

        std::fs::remove_file(save_path).unwrap();
    }
    run_test!(hoge, hoge_cpu, hoge_nvidia);
}

# ZeNu - A Deep Learning Library for Rust

ZeNu is a simple and intuitive deep learning library written in Rust. It provides the building blocks for creating and training neural networks, with a focus on ease of use and flexibility.

ZeNu comes from 冒頓単于(bokutotuzennu)

**Please note that ZeNu is currently under active development and may undergo significant changes.**

## Features

- Autograd engine for automatic differentiation
- Tensor operations and linear algebra utilities
- Neural network layers and model definition
- Optimizers for training models
- Modular design for easy extensibility

## Installation

To use ZeNu in your Rust project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
zenu = "0.1.0"
```

## Getting Started

Here's a simple example of defining and training a model using ZeNu:

```rust
use zenu::{
    dataset::{train_val_split, DataLoader, Dataset},
    mnist::minist_dataset,
    update_parameters, Model,
};
use zenu_autograd::{
    creator::from_vec::from_vec,
    functions::{activation::sigmoid::sigmoid, loss::cross_entropy::cross_entropy},
    Variable,
};
use zenu_layer::{layers::linear::Linear, Layer};
use zenu_matrix::{
    matrix::{IndexItem, ToViewMatrix},
    operation::max::MaxIdx,
};
use zenu_optimizer::sgd::SGD;

struct SingleLayerModel {
    linear: Linear<f32>,
}

impl SingleLayerModel {
    fn new() -> Self {
        let mut linear = Linear::new(784, 10);
        linear.init_parameters(None);
        Self { linear }
    }
}

impl Model<f32> for SingleLayerModel {
    fn predict(&self, inputs: &[Variable<f32>]) -> Variable<f32> {
        let x = &inputs[0];
        let x = self.linear.call(x.clone());
        sigmoid(x)
    }
}

struct MnistDataset {
    data: Vec<(Vec<u8>, u8)>,
}

impl Dataset<f32> for MnistDataset {
    type Item = (Vec<u8>, u8);

    fn item(&self, item: usize) -> Vec<Variable<f32>> {
        let (x, y) = &self.data[item];
        let x_f32 = x.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let x = from_vec(x_f32, [784]);
        let y_onehot = (0..10)
            .map(|i| if i == *y as usize { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let y = from_vec(y_onehot, [10]);
        vec![x, y]
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn all_data(&mut self) -> &mut [Self::Item] {
        &mut self.data as &mut [Self::Item]
    }
}

fn main() {
    let (train, test) = minist_dataset().unwrap();
    let (train, val) = train_val_split(&train, 0.8, true);

    let test_dataloader = DataLoader::new(MnistDataset { data: test }, 1);

    let sgd = SGD::new(0.01);
    let model = SingleLayerModel::new();

    for epoch in 0..10 {
        let mut train_dataloader = DataLoader::new(
            MnistDataset {
                data: train.clone(),
            },
            16,
        );
        let val_dataloader = DataLoader::new(MnistDataset { data: val.clone() }, 16);

        train_dataloader.shuffle();

        let mut epoch_loss_train: f32 = 0.;
        let mut num_iter_train = 0;
        for batch in train_dataloader {
            let input = batch[0].clone();
            let target = batch[1].clone();
            let y_pred = model.predict(&[input]);
            let loss = cross_entropy(y_pred, target);
            update_parameters(loss.clone(), &sgd);
            epoch_loss_train += loss.get_data().index_item([]);
            num_iter_train += 1;
        }

        let mut epoch_loss_val = 0.;
        let mut num_iter_val = 0;
        for batch in val_dataloader {
            let input = batch[0].clone();
            let target = batch[1].clone();
            let y_pred = model.predict(&[input]);
            let loss = cross_entropy(y_pred, target);
            epoch_loss_val += loss.get_data().index_item([]);
            num_iter_val += 1;
        }

        println!(
            "Epoch: {}, Train Loss: {}, Val Loss: {}",
            epoch,
            epoch_loss_train / num_iter_train as f32,
            epoch_loss_val / num_iter_val as f32
        );
    }

    let mut test_loss = 0.;
    let mut num_iter_test = 0;
    let mut correct = 0;
    let mut total = 0;
    for batch in test_dataloader {
        let input = batch[0].clone();
        let target = batch[1].clone();
        let y_pred = model.predict(&[input]);
        let loss = cross_entropy(y_pred.clone(), target.clone());
        test_loss += loss.get_data().index_item([]);
        num_iter_test += 1;
        let y_pred = y_pred.get_data();
        let max_idx = y_pred.to_view().max_idx()[0];
        let target = target.get_data();
        let target = target.to_view().max_idx()[0];
        if max_idx == target {
            correct += 1;
        }
        total += 1;
    }

    println!("Accuracy: {}", correct as f32 / total as f32);

    println!("Test Loss: {}", test_loss / num_iter_test as f32);
}

```

## Contributing

Contributions to ZeNu are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/bokutotu/zenu).

## License

ZeNu is licensed under the [MIT License](LICENSE).

Please keep in mind that ZeNu is currently in the early stages of development, and the API may change as the project evolves.

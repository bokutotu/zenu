# Zenu

Zenu is a simple and intuitive deep learning library written in Rust. It provides the building blocks for creating and training neural networks, with a focus on ease of use and flexibility.

## Features

- High-level API for defining and training neural networks
- Integration with popular datasets like MNIST
- Modular design for easy extensibility
- Efficient computation using the underlying zenu-matrix and zenu-autograd libraries

## Getting Started

To use Zenu in your Rust project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
zenu = "0.1.0"
```

Here's a simple example of defining and training a model using Zenu:

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

// Define your model
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

// Define your dataset
struct MnistDataset {
    data: Vec<(Vec<u8>, u8)>,
}

impl Dataset<f32> for MnistDataset {
    type Item = (Vec<u8>, u8);

    fn item(&self, item: usize) -> Vec<Variable<f32>> {
        // ... Implement your dataset logic
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn all_data(&mut self) -> &mut [Self::Item] {
        &mut self.data as &mut [Self::Item]
    }
}

fn main() {
    // Load and prepare your data
    let (train, test) = minist_dataset().unwrap();
    let (train, val) = train_val_split(&train, 0.8, true);

    let test_dataloader = DataLoader::new(MnistDataset { data: test }, 1);

    // Create your model and optimizer
    let sgd = SGD::new(0.01);
    let model = SingleLayerModel::new();

    // Train your model
    for epoch in 0..10 {
        // ... Implement your training loop
    }

    // Evaluate your model
    let mut test_loss = 0.;
    let mut num_iter_test = 0;
    let mut correct = 0;
    let mut total = 0;
    for batch in test_dataloader {
        // ... Implement your evaluation logic
    }

    println!("Accuracy: {}", correct as f32 / total as f32);
    println!("Test Loss: {}", test_loss / num_iter_test as f32);
}
```

For more details and examples, please refer to the [documentation](https://docs.rs/zenu).

## License

Zenu is licensed under the [MIT License](LICENSE).


# ZeNu

**ZeNu** is a high-performance deep learning framework implemented in pure Rust. It features an intuitive API and high extensibility.

## Features

- 🦀 **Pure Rust implementation**: Maximizes safety and performance
- ⚡ **GPU performance**: Comparable to PyTorch (supports CUDA 12.3 + cuDNN 9)
- 🔧 **Simple and intuitive API**
- 📦 **Modular design**: Easy to extend

## Installation

Add the following to your Cargo.toml:

```toml
[dependencies]
zenu = "0.1"

# To enable CUDA support:
[dependencies.zenu]
version = "0.1"
features = ["nvidia"]
```

## Supported Features

### Layers
- Linear
- Convolution 2D
- Batch Normalization 2D
- LSTM
- RNN
- GRU
- MaxPool 2D
- Dropout

### Optimizers
- SGD
- Adam
- AdamW

### Device Support
- CPU
- CUDA (NVIDIA GPU)
  - CUDA 12.3
  - cuDNN 9

## Project Structure

```
zenu/
├── zenu               # Main library
├── zenu-autograd      # Automatic differentiation engine
├── zenu-layer         # Neural network layers
├── zenu-matrix        # Matrix operations
├── zenu-optimizer     # Optimization algorithms
├── zenu-cuda          # CUDA implementation
└── Other support crates
```

## Examples

Check the `examples/` directory for detailed implementations:
- MNIST classification
- CIFAR10 classification
- ResNet implementation

### Simple Usage Example

Here is a simple example of defining and training a model using ZeNu:

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

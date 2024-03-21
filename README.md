# Zenu - A Deep Learning Library for Rust

Zenu is a simple and intuitive deep learning library written in Rust. It provides the building blocks for creating and training neural networks, with a focus on ease of use and flexibility.

**Please note that Zenu is currently under active development and may undergo significant changes.**

## Features

- Autograd engine for automatic differentiation
- Tensor operations and linear algebra utilities
- Neural network layers and model definition
- Optimizers for training models
- Modular design for easy extensibility

## Installation

To use Zenu in your Rust project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
zenu = "0.1.0"
```

## Getting Started

Here's a simple example of defining and training a model using Zenu:

```rust
use zenu::Model;
use zenu::layers::{Linear, ReLU};
use zenu::optim::SGD;

struct Net {
    l1: Linear,
    l2: Linear,
}

impl Net {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Net {
            l1: Linear::new(input_size, hidden_size),
            l2: Linear::new(hidden_size, output_size),
        }
    }
}

impl Model for Net {
    fn forward(&self, inputs: &Variable) -> Variable {
        let x = self.l1.forward(inputs);
        let x = ReLU::forward(&x);
        self.l2.forward(&x)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.l1.parameters());
        params.extend(self.l2.parameters());
        params
    }
}

fn main() {
    let input_size = 784;
    let hidden_size = 128;
    let output_size = 10;
    let model = Net::new(input_size, hidden_size, output_size);

    let mut optimizer = SGD::new(0.01);

    // Training loop
    for epoch in 0..10 {
        // ...
        let loss = // Compute loss
        loss.backward();
        optimizer.step(&model.parameters());
    }
}
```

## Contributing

Contributions to Zenu are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/bokutotu/zenu).

## License

Zenu is licensed under the [MIT License](LICENSE).

Please keep in mind that Zenu is currently in the early stages of development, and the API may change as the project evolves.

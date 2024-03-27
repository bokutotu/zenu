# ZeNu Optimizer

ZeNu Optimizer is a collection of optimization algorithms for training neural networks. It provides various optimizers that can be used with the ZeNu deep learning library.

## Features

- Stochastic Gradient Descent (SGD) optimizer
- Integration with ZeNu Autograd for gradient computation
- Easy integration with ZeNu models and layers

## Getting Started

To use ZeNu Optimizer in your Rust project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
zenu-optimizer = "0.1.0"
```

Here's a simple example of using the SGD optimizer from ZeNu Optimizer:

```rust
use zenu_autograd::{creator::from_vec::from_vec, Variable};
use zenu_optimizer::sgd::SGD;

fn main() {
    let variable = from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]);
    variable.set_grad(from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]));

    let sgd = SGD::new(0.01);
    sgd.update(&[variable.clone()]);

    // The variable has been updated by the optimizer
    // Perform further computations with the updated variable
}
```

For more details and examples, please refer to the [documentation](https://docs.rs/zenu-optimizer).

## License

ZeNu Optimizer is licensed under the [MIT License](LICENSE).

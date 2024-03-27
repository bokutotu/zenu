# ZeNu Autograd

ZeNu Autograd is an automatic differentiation library for Rust. It provides the foundation for building and training neural networks by automatically computing gradients of mathematical expressions.

## Features

- Define and manipulate mathematical expressions using Variables
- Automatically compute gradients through reverse-mode automatic differentiation
- Support for various mathematical operations and functions
- Integration with ZeNu deep learning library

## Getting Started

To use ZeNu Autograd in your Rust project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
zenu-autograd = "0.1.0"
```

Here's a simple example of using ZeNu Autograd:

```rust
use zenu_autograd::{Variable, creator::from_vec::from_vec};

fn main() {
    let x = from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]);
    let y = from_vec(vec![7., 8., 9., 10., 11., 12.], [3, 2]);
    let z = x.clone() * y.clone() + y.clone();

    z.backward();

    let x_grad = x.get_grad().unwrap();
    let y_grad = y.get_grad().unwrap();

    // Perform further computations with the gradients
}
```

For more details and examples, please refer to the [documentation](https://docs.rs/zenu-autograd).

## License

ZeNu Autograd is licensed under the [MIT License](LICENSE).

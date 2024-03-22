# Zenu Layer

Zenu Layer is a collection of neural network layers implemented in Rust. It provides building blocks for constructing neural networks and integrates with the Zenu deep learning library.

## Features

- Various layer types, including fully connected (linear) layers
- Layer parameter initialization
- Forward pass computation
- Integration with Zenu Autograd for automatic differentiation

## Getting Started

To use Zenu Layer in your Rust project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
zenu-layer = "0.1.0"
```

Here's a simple example of using a linear layer from Zenu Layer:

```rust
use zenu_autograd::creator::from_vec::from_vec;
use zenu_layer::layers::linear::Linear;
use zenu_layer::Layer;

fn main() {
    // Create a new linear layer with input dimension 3 and output dimension 2
    let mut linear_layer = Linear::new(3, 2);

    // Initialize the layer parameters with a random seed
    linear_layer.init_parameters(Some(42));

    // Create input data as a Variable
    let input = from_vec(vec![1., 2., 3.], [1, 3]);

    // Perform a forward pass through the layer
    let output = linear_layer.call(input);

    // Access the layer parameters
    let parameters = linear_layer.parameters();
}
```

For more details and examples, please refer to the [documentation](https://docs.rs/zenu-layer).

## License

Zenu Layer is licensed under the [MIT License](LICENSE).

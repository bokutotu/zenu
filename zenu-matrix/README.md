# Zenu Matrix

Zenu Matrix is a linear algebra library for Rust, providing efficient matrix operations and various utilities for working with matrices.

## Features

- Matrix creation, indexing, and slicing
- Element-wise operations
- Matrix multiplication (GEMM)
- Transposition
- Broadcasting
- Random matrix generation
- Integration with BLAS for optimized performance

## Getting Started

To use Zenu Matrix in your Rust project, add the following to your `Cargo.toml` file:

```toml
[dependencies]
zenu-matrix = "0.1.0"
```

Here's a simple example of using Zenu Matrix:

```rust
use zenu_matrix::{
    matrix::{IndexItem, OwnedMatrix},
    matrix_impl::OwnedMatrixDyn,
    operation::asum::Asum,
};

fn main() {
    let a = OwnedMatrixDyn::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
    let b = OwnedMatrixDyn::from_vec(vec![7., 8., 9., 10., 11., 12.], [3, 2]);
    let c = a.clone() * b.clone();

    assert_eq!(c.index_item([0, 0]), 58.);
    assert_eq!(c.index_item([0, 1]), 64.);
    assert_eq!(c.index_item([1, 0]), 139.);
    assert_eq!(c.index_item([1, 1]), 154.);
}
```

For more details and examples, please refer to the [documentation](https://docs.rs/zenu-matrix).

## License

Zenu Matrix is licensed under the [MIT License](LICENSE).

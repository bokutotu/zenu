# ZeNu Matrix

ZeNu Matrix is a high-performance linear algebra library for Rust, designed to provide efficient matrix operations and various utilities for working with matrices. Whether you are building complex machine learning models or performing scientific computations, ZeNu Matrix offers the tools you need.

## Features

- **Comprehensive Matrix Operations**: Create, index, and slice matrices with ease.
- **Element-wise Operations**: Perform operations on individual elements or entire matrices.
- **Efficient Matrix Multiplication (GEMM)**: Utilize optimized routines for matrix multiplication.
- **Matrix Transposition**: Quickly transpose matrices as needed.
- **Broadcasting**: Seamlessly broadcast operations over matrices of different shapes.
- **Random Matrix Generation**: Generate matrices with random values for testing and initialization.
- **BLAS Integration**: Leverage BLAS for optimized performance on supported hardware.
- **CUDA Support**: Accelerate computations using NVIDIA GPUs with CUDA integration.

## Getting Started

To start using ZeNu Matrix, add it to your `Cargo.toml`:

```toml
[dependencies]
zenu-matrix = "0.1.1"
```

### Example

Here's a simple example of using ZeNu Matrix:

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

ZeNu Matrix is licensed under the [MIT License](LICENSE).

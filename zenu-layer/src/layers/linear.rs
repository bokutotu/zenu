//! # Linear Layer
//!
//! This module provides an implementation of a linear layer for a neural network.
//! The linear layer performs a linear transformation on the input data using a weight matrix and a bias vector.
//!
//! ## Usage
//!
//! To create a new linear layer, use the `Linear::new(in_dim, out_dim)` function, specifying the input and output dimensions.
//! The layer parameters (weight and bias) can be initialized using the `init_parameters(&mut self, seed: Option<u64>)` method.
//!
//! To use the linear layer in a forward pass, call the `call(&self, input: Variable<T>) -> Variable<T>` method, passing the input data as a `Variable`.
//! The output of the layer will be returned as a new `Variable`.
//!
//! To access the layer parameters, use the `parameters(&self) -> Vec<Variable<T>>` method, which returns a vector of the weight and bias variables.
//! To load pre-trained parameters into the layer, use the `load_parameters(&mut self, parameters: &[Variable<T>])` method, passing a slice of variables.
//!
//! ## Example
//!
//! ```rust
//! use zenu_autograd::creator::from_vec::from_vec;
//! use zenu_layer::layers::linear::Linear;
//! use zenu_layer::Layer;
//!
//! // Create a new linear layer with input dimension 3 and output dimension 2
//! let mut linear_layer = Linear::new(3, 2);
//!
//! // Initialize the layer parameters with a random seed
//! linear_layer.init_parameters(Some(42));
//!
//! // Create input data as a Variable
//! let input = from_vec(vec![1., 2., 3.], [1, 3]);
//!
//! // Perform a forward pass through the layer
//! let output = linear_layer.call(input);
//!
//! // Access the layer parameters
//! let parameters = linear_layer.parameters();
//! ```
//!
//! ## Layer Trait Implementation
//!
//! The `Linear` struct implements the `Layer` trait, which defines the following methods:
//!
//! - `init_parameters(&mut self, seed: Option<u64>)`: Initializes the layer parameters (weight and bias) randomly or with a specified seed.
//! - `parameters(&self) -> Vec<Variable<T>>`: Returns a vector of the layer parameters as `Variable`s.
//! - `load_parameters(&mut self, parameters: &[Variable<T>])`: Loads pre-trained parameters into the layer.
//! - `call(&self, input: Variable<T>) -> Variable<T>`: Performs a forward pass through the layer, applying the linear transformation to the input.
//! - `shape_check(&self, input: &Variable<T>)`: Checks the shape of the input data to ensure compatibility with the layer.
//!
//! ## Type Parameters
//!
//! - `T`: The numeric type used for the layer parameters and computations. It must implement the `Num` trait.
//!
//! ## Fields
//!
//! - `in_dim`: The input dimension of the layer.
//! - `out_dim`: The output dimension of the layer.
//! - `weight`: The weight matrix of the layer, stored as an `Option<Variable<T>>`.
//! - `bias`: The bias vector of the layer, stored as an `Option<Variable<T>>`.

use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::{
    creator::{rand::normal, zeros::zeros},
    functions::matmul::matmul,
    Variable,
};
use zenu_matrix::{device::Device, num::Num};

use crate::Layer;

/// A linear layer in a neural network.
pub struct Linear<T: Num, D: Device> {
    in_dim: usize,
    out_dim: usize,
    weight: Option<Variable<T, D>>,
    bias: Option<Variable<T, D>>,
}

impl<T: Num, D: Device> Linear<T, D> {
    /// Creates a new linear layer with the specified input and output dimensions.
    ///
    /// # Arguments
    ///
    /// * `in_dim` - The input dimension of the layer.
    /// * `out_dim` - The output dimension of the layer.
    ///
    /// # Returns
    ///
    /// A new `Linear` instance.
    #[must_use]
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Linear {
            in_dim,
            out_dim,
            weight: None,
            bias: None,
        }
    }
}

impl<T: Num> Layer<T> for Linear<T> {
    /// Initializes the layer parameters (weight and bias) randomly or with a specified seed.
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional seed value for reproducible parameter initialization.
    fn init_parameters(&mut self, seed: Option<u64>)
    where
        StandardNormal: Distribution<T>,
    {
        let bias = zeros([self.out_dim]);
        bias.set_is_train(true);
        self.bias = Some(bias);
        let weight = normal(T::zero(), T::one(), seed, [self.in_dim, self.out_dim]);
        weight.set_is_train(true);
        self.weight = Some(weight);
    }

    /// Returns a vector of the layer parameters as `Variable`s.
    ///
    /// # Returns
    ///
    /// A vector containing the weight and bias variables of the layer.
    fn parameters(&self) -> Vec<Variable<T>> {
        vec![self.weight.clone().unwrap(), self.bias.clone().unwrap()]
    }

    /// Loads pre-trained parameters into the layer.
    ///
    /// # Arguments
    ///
    /// * `parameters` - A slice of `Variable`s containing the weight and bias parameters.
    ///
    /// # Panics
    ///
    /// Panics if the number of parameters is not 2, or if the shapes of the parameters are invalid.
    fn load_parameters(&mut self, parameters: &[Variable<T>]) {
        assert_eq!(parameters.len(), 2, "parameters must be 2");
        assert_eq!(
            parameters[0].get_data().shape().len(),
            2,
            "weight must be 2D"
        );
        assert_eq!(parameters[1].get_data().shape().len(), 1, "bias must be 1D");
        self.weight = Some(parameters[0].clone());
        self.bias = Some(parameters[1].clone());
    }

    /// Performs a forward pass through the layer, applying the linear transformation to the input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data as a `Variable`.
    ///
    /// # Returns
    ///
    /// The output of the layer as a new `Variable`.
    ///
    /// # Panics
    ///
    /// Panics if the weight or bias parameters are not initialized.
    fn call(&self, input: Variable<T>) -> Variable<T> {
        assert!(self.weight.is_some(), "weight is not initialized");
        assert!(self.bias.is_some(), "bias is not initialized");
        let weight = self.weight.clone().unwrap();
        let bias = self.bias.clone().unwrap();

        self.shape_check(&input);

        let output = matmul(input, weight);
        output + bias
    }

    /// Checks the shape of the input data to ensure compatibility with the layer.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data as a reference to a `Variable`.
    ///
    /// # Panics
    ///
    /// Panics if the input shape is not 2D or if the input dimension does not match the layer's input dimension.
    fn shape_check(&self, input: &Variable<T>) {
        assert_eq!(input.get_data().shape().len(), 2, "input shape must be 2D");
        assert_eq!(
            input.get_data().shape()[1],
            self.in_dim,
            "input shape must be (batch_size, {})",
            self.in_dim
        );
    }
}

#[cfg(test)]
mod tests {
    use zenu_autograd::creator::from_vec::from_vec;
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use crate::Layer;

    use super::Linear;

    #[test]
    fn linear_batch_size_1() {
        let weight = from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]);
        let bias = from_vec(vec![1., 2.], [2]);
        let mut linear_layer = Linear::new(3, 2);
        linear_layer.load_parameters(&[weight.clone(), bias.clone()]);
        let x = from_vec(vec![1., 2., 3.], [1, 3]);
        let y = linear_layer.call(x);
        y.backward();
        let y_data = y.get_data();
        let ans = OwnedMatrixDyn::from_vec(vec![23., 30.], [1, 2]);
        let diff = y_data - ans;
        assert!(diff.asum() < 1e-6);
        let bias_grad = bias.get_grad().unwrap().get_data();
        let weight_grad = weight.get_grad().unwrap().get_data();
        let bias_ans = OwnedMatrixDyn::from_vec(vec![1., 1.], [2]);
        let weight_ans = OwnedMatrixDyn::from_vec(vec![1., 1., 2., 2., 3., 3.], [3, 2]);
        let diff_bias = bias_grad - bias_ans;
        let diff_weight = weight_grad - weight_ans;
        let diff_bias_asum = diff_bias.asum();
        let diff_weight_asum = diff_weight.asum();
        assert!(diff_bias_asum < 1e-6);
        assert!(diff_weight_asum < 1e-6);
    }

    #[test]
    fn linear_batch_size_16() {
        let weight = from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]);
        let bias = from_vec(vec![1., 2.], [2]);
        let mut linear_layer = Linear::new(3, 2);
        linear_layer.load_parameters(&[weight.clone(), bias.clone()]);
        let mut vec = Vec::new();
        for i in 1..=16 * 3 {
            vec.push(i as f64);
        }
        let x = from_vec(vec, [16, 3]);
        let y = linear_layer.call(x);
        y.backward();
        let y_data = y.get_data();
        let y_ans = OwnedMatrixDyn::from_vec(
            vec![
                23., 30., 50., 66., 77., 102., 104., 138., 131., 174., 158., 210., 185., 246.,
                212., 282., 239., 318., 266., 354., 293., 390., 320., 426., 347., 462., 374., 498.,
                401., 534., 428., 570.,
            ],
            [16, 2],
        );
        let diff = y_data - y_ans;
        assert!(diff.asum() < 1e-6);
        let bias_grad = bias.get_grad().unwrap().get_data();
        let weight_grad = weight.get_grad().unwrap().get_data();
        let bias_ans = OwnedMatrixDyn::from_vec(vec![16., 16.], [2]);
        let weight_ans = OwnedMatrixDyn::from_vec(vec![376., 376., 392., 392., 408., 408.], [3, 2]);
        let diff_bias = bias_grad - bias_ans;
        let diff_weight = weight_grad - weight_ans;
        let diff_bias_asum = diff_bias.asum();
        let diff_weight_asum = diff_weight.asum();
        assert!(diff_bias_asum < 1e-6);
        assert!(diff_weight_asum < 1e-6);
    }
}

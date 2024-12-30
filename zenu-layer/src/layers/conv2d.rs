// use std::{cell::RefCell, collections::HashMap};
//
// use rand_distr::{Distribution, StandardNormal};
// use zenu_autograd::{
//     creator::{rand::normal, zeros::zeros},
//     nn::conv2d::{conv2d, Conv2dConfigs},
//     Variable,
// };
// use zenu_matrix::{device::Device, dim::DimTrait, nn::conv2d::conv2d_out_size, num::Num};
//
// use crate::{Module, Parameters};
//
// pub struct Conv2d<T: Num, D: Device> {
//     pub filter: Variable<T, D>,
//     pub bias: Option<Variable<T, D>>,
//     config: RefCell<Option<Conv2dConfigs<T>>>,
//     stride: (usize, usize),
//     padding: (usize, usize),
// }
//
// impl<T: Num, D: Device> Module<T, D> for Conv2d<T, D> {
//     type Input = Variable<T, D>;
//     type Output = Variable<T, D>;
//     fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
//         if self.config.borrow().is_none() {
//             let input_shape = input.get_data().shape();
//             let filter_shape = self.filter.get_data().shape();
//             let output_shape = conv2d_out_size(
//                 input_shape.slice(),
//                 filter_shape.slice(),
//                 self.padding,
//                 self.stride,
//             );
//             let config = Conv2dConfigs::new(
//                 input_shape,
//                 output_shape.into(),
//                 filter_shape,
//                 self.stride,
//                 self.padding,
//                 20,
//             );
//             *self.config.borrow_mut() = Some(config);
//         }
//         conv2d(
//             input,
//             self.filter.clone(),
//             self.stride,
//             self.padding,
//             self.bias.clone(),
//             Some(self.config.borrow().as_ref().unwrap().clone()),
//         )
//     }
// }
//
// impl<T: Num, D: Device> Parameters<T, D> for Conv2d<T, D> {
//     fn weights(&self) -> HashMap<String, Variable<T, D>> {
//         HashMap::new()
//             .into_iter()
//             .chain(std::iter::once((
//                 String::from("conv2d.filter"),
//                 self.filter.clone(),
//             )))
//             .collect()
//     }
//
//     fn biases(&self) -> HashMap<String, Variable<T, D>> {
//         self.bias
//             .as_ref()
//             .map(|bias| {
//                 HashMap::new()
//                     .into_iter()
//                     .chain(std::iter::once((String::from("conv2d.bias"), bias.clone())))
//                     .collect()
//             })
//             .unwrap_or_default()
//     }
// }
//
// impl<T: Num, D: Device> Conv2d<T, D> {
//     #[must_use]
//     pub fn new(
//         input_channel: usize,
//         output_channel: usize,
//         kernel_size: (usize, usize),
//         stride: (usize, usize),
//         padding: (usize, usize),
//         bias: bool,
//     ) -> Self
//     where
//         StandardNormal: Distribution<T>,
//     {
//         let filter_shape = [output_channel, input_channel, kernel_size.0, kernel_size.1];
//         let bias = if bias {
//             let bias = zeros([1, output_channel, 1, 1]);
//             bias.set_is_train(true);
//             bias.set_name("conv2d.bias");
//             Some(bias)
//         } else {
//             None
//         };
//         let filter = normal(T::zero(), T::one(), None, filter_shape);
//
//         filter.set_is_train(true);
//         filter.set_name("conv2d.filter");
//
//         Conv2d {
//             filter,
//             bias,
//             config: RefCell::new(None),
//             stride,
//             padding,
//         }
//     }
//
//     pub fn to<Dout: Device>(self) -> Conv2d<T, Dout> {
//         Conv2d {
//             filter: self.filter.to(),
//             bias: self.bias.map(|b| b.to()),
//             config: RefCell::new(None),
//             stride: self.stride,
//             padding: self.padding,
//         }
//     }
// }

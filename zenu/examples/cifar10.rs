// use zenu::{
//     autograd::{
//         creator::from_vec::from_vec,
//         functions::{activation::relu::relu, flatten::flatten, loss::cross_entropy::cross_entropy},
//         no_train, set_train, Variable,
//     },
//     dataset::{train_val_split, DataLoader, Dataset},
//     dataset_loader::cifar10_dataset,
//     layer::{
//         layers::{batch_norm_2d::BatchNorm2d, conv2d::Conv2d, linear::Linear},
//         Module,
//     },
//     matrix::device::{cpu::Cpu, nvidia::Nvidia, Device},
//     optimizer::sgd::SGD,
//     update_parameters,
// };
//
// struct ConvNet<D: Device> {
//     conv1: Conv2d<f32, D>,
//     batch_norm1: BatchNorm2d<f32, D>,
//     conv2: Conv2d<f32, D>,
//     batch_norm2: BatchNorm2d<f32, D>,
//     linear1: Linear<f32, D>,
//     linear2: Linear<f32, D>,
// }
//
// impl<D: Device> ConvNet<D> {
//     fn new() -> Self {
//         let conv1 = Conv2d::new(3, 32, (3, 3), (1, 1), (1, 1), true);
//         let batch_norm1 = BatchNorm2d::new(32, 0.9);
//         let conv2 = Conv2d::new(32, 64, (3, 3), (1, 1), (1, 1), true);
//         let batch_norm2 = BatchNorm2d::new(64, 0.9);
//         let linear1 = Linear::new(65536, 512, true);
//         let linear2 = Linear::new(512, 10, true);
//         Self {
//             conv1,
//             batch_norm1,
//             conv2,
//             batch_norm2,
//             linear1,
//             linear2,
//         }
//     }
//
//     fn to<Dout: Device>(self) -> ConvNet<Dout> {
//         ConvNet {
//             conv1: self.conv1.to(),
//             batch_norm1: self.batch_norm1.to(),
//             conv2: self.conv2.to(),
//             batch_norm2: self.batch_norm2.to(),
//             linear1: self.linear1.to(),
//             linear2: self.linear2.to(),
//         }
//     }
// }
//
// impl<D: Device> Module<f32, D> for ConvNet<D> {
//     fn call(&self, inputs: Variable<f32, D>) -> Variable<f32, D> {
//         let x = &inputs;
//         let x = self.conv1.call(x.clone());
//         let x = self.batch_norm1.call(x);
//         let x = relu(x);
//         let x = self.conv2.call(x);
//         let x = self.batch_norm2.call(x);
//         let x = relu(x);
//         let x = flatten(x);
//         let x = self.linear1.call(x);
//         let x = relu(x);
//         self.linear2.call(x)
//     }
// }
//
// struct CiFar10Dataset {
//     data: Vec<(Vec<u8>, u8)>,
// }
//
// impl Dataset<f32> for CiFar10Dataset {
//     type Item = (Vec<u8>, u8);
//
//     fn item(&self, index: usize) -> Vec<Variable<f32, Cpu>> {
//         let (x, y) = &self.data[index];
//         let x_f32 = x.iter().map(|&x| x as f32).collect::<Vec<_>>();
//         let x = from_vec(x_f32, [3, 32, 32]);
//         let y_onehot = (0..10)
//             .map(|i| if i == *y as usize { 1.0 } else { 0.0 })
//             .collect::<Vec<_>>();
//         let y = from_vec(y_onehot, [10]);
//         vec![x, y]
//     }
//
//     fn len(&self) -> usize {
//         self.data.len()
//     }
//
//     fn all_data(&mut self) -> &mut [Self::Item] {
//         &mut self.data as &mut [Self::Item]
//     }
// }
//
// fn main() {
//     let (train, test) = cifar10_dataset().unwrap();
//     let (train, val) = train_val_split(&train, 0.8, true);
//
//     let test_dataloader = DataLoader::new(CiFar10Dataset { data: test }, 1);
//
//     let sgd = SGD::new(0.01);
//     let model = ConvNet::<Cpu>::new();
//
//     let model = model.to::<Nvidia>();
//
//     for epoch in 0..10 {
//         let mut train_dataloader = DataLoader::new(
//             CiFar10Dataset {
//                 data: train.clone(),
//             },
//             512,
//         );
//         let val_dataloader = DataLoader::new(CiFar10Dataset { data: val.clone() }, 1);
//
//         train_dataloader.shuffle();
//
//         let mut epoch_loss_train: f32 = 0.;
//         let mut num_iter_train = 0;
//
//         for batch in train_dataloader {
//             let x = batch[0].clone();
//             let y = batch[1].clone();
//             if x.get_shape()[0] != 512 {
//                 continue;
//             }
//             let x = x.to::<Nvidia>();
//             let y = y.to::<Nvidia>();
//             let output = model.call(x);
//             let loss = cross_entropy(output, y);
//             let loss_itm = loss.get_data().index_item([]);
//             epoch_loss_train += loss_itm;
//             num_iter_train += 1;
//             update_parameters(loss, &sgd);
//         }
//
//         let mut epoch_loss_val: f32 = 0.;
//         let mut num_iter_val = 0;
//
//         no_train();
//         for batch in val_dataloader {
//             let x = batch[0].clone();
//             let y = batch[1].clone();
//             let x = x.to::<Nvidia>();
//             let y = y.to::<Nvidia>();
//             let output = model.call(x);
//             let loss = cross_entropy(output, y);
//             epoch_loss_val += loss.get_data().index_item([]);
//             num_iter_val += 1;
//         }
//         set_train();
//
//         println!(
//             "Epoch: {}, Train Loss: {}, Val Loss: {}",
//             epoch,
//             epoch_loss_train / num_iter_train as f32,
//             epoch_loss_val / num_iter_val as f32
//         );
//     }
//
//     let mut test_loss = 0.;
//     let mut num_iter_test = 0;
//     let mut correct = 0;
//     let mut total = 0;
//     for batch in test_dataloader {
//         let x = batch[0].clone();
//         let y = batch[1].clone();
//         let x = x.to::<Nvidia>();
//         let y = y.to::<Nvidia>();
//         let output = model.call(x);
//         let loss = cross_entropy(output.clone(), y.clone());
//         test_loss += loss.get_data().index_item([]);
//         num_iter_test += 1;
//         let y_pred = output.get_data().to_ref().max_idx()[0];
//         let y_true = y.get_data().to_ref().max_idx()[0];
//         correct += (y_pred == y_true) as usize;
//         total += 1;
//     }
//
//     println!(
//         "Test Loss: {}, Test Accuracy: {}",
//         test_loss / num_iter_test as f32,
//         correct as f32 / total as f32
//     );
// }
fn main() {
    println!("here");
}

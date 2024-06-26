fn main() {
    println!("Hello, world!");
}
// use zenu::{
//     dataset::{train_val_split, DataLoader, Dataset},
//     dataset_loader::cifar10_dataset,
//     update_parameters, Model,
// };
// use zenu_autograd::{
//     creator::from_vec::from_vec,
//     functions::{activation::relu::relu, flatten::flatten, loss::cross_entropy::cross_entropy},
//     no_train, set_train, Variable,
// };
// use zenu_layer::{
//     layers::{batch_norm::BatchNorm, conv2d::Conv2d, linear::Linear},
//     Layer,
// };
// use zenu_matrix::{matrix::IndexItem, matrix::ToViewMatrix, operation::max::MaxIdx};
// use zenu_optimizer::sgd::SGD;
//
// struct ConvNet {
//     conv1: Conv2d<f32>,
//     batch_norm1: BatchNorm<f32>,
//     conv2: Conv2d<f32>,
//     batch_norm2: BatchNorm<f32>,
//     linear1: Linear<f32>,
//     linear2: Linear<f32>,
// }
//
// impl ConvNet {
//     fn new() -> Self {
//         let mut conv1 = Conv2d::new(3, 32, (3, 3), (1, 1), (1, 1), true);
//         let mut batch_norm1 = BatchNorm::new(32, 0.9, 1e-5);
//         let mut conv2 = Conv2d::new(32, 64, (3, 3), (1, 1), (1, 1), true);
//         let mut batch_norm2 = BatchNorm::new(64, 0.9, 1e-5);
//         let mut linear1 = Linear::new(65536, 512);
//         let mut linear2 = Linear::new(512, 10);
//         conv1.init_parameters(None);
//         batch_norm1.init_parameters(None);
//         conv2.init_parameters(None);
//         batch_norm2.init_parameters(None);
//         linear1.init_parameters(None);
//         linear2.init_parameters(None);
//         Self {
//             conv1,
//             batch_norm1,
//             conv2,
//             batch_norm2,
//             linear1,
//             linear2,
//         }
//     }
// }
//
// impl Model<f32> for ConvNet {
//     fn predict(&self, inputs: &[Variable<f32>]) -> Variable<f32> {
//         let x = &inputs[0];
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
//     fn item(&self, index: usize) -> Vec<Variable<f32>> {
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
//     let model = ConvNet::new();
//
//     for epoch in 0..10 {
//         let mut train_dataloader = DataLoader::new(
//             CiFar10Dataset {
//                 data: train.clone(),
//             },
//             16,
//         );
//         let val_dataloader = DataLoader::new(CiFar10Dataset { data: val.clone() }, 16);
//
//         train_dataloader.shuffle();
//
//         let mut epoch_loss_train: f32 = 0.;
//         let mut num_iter_train = 0;
//
//         for batch in train_dataloader {
//             let x = batch[0].clone();
//             let y = batch[1].clone();
//             let output = model.predict(&[x]);
//             let loss = cross_entropy(output, y);
//             epoch_loss_train += loss.get_data().index_item([]);
//             num_iter_train += 1;
//             update_parameters(loss.clone(), &sgd);
//         }
//
//         let mut epoch_loss_val: f32 = 0.;
//         let mut num_iter_val = 0;
//
//         no_train();
//         for batch in val_dataloader {
//             let x = batch[0].clone();
//             let y = batch[1].clone();
//             let output = model.predict(&[x]);
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
//         let output = model.predict(&[x]);
//         let loss = cross_entropy(output.clone(), y.clone());
//         test_loss += loss.get_data().index_item([]);
//         num_iter_test += 1;
//         let y_pred = output.get_data().to_view().max_idx()[0];
//         let y_true = y.get_data().to_view().max_idx()[0];
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

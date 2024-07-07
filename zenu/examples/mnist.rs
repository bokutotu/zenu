use serde::{Deserialize, Serialize};
use zenu::{
    dataset::{train_val_split, DataLoader, Dataset},
    dataset_loader::mnist_dataset,
    update_parameters,
};
use zenu_autograd::{
    creator::from_vec::from_vec,
    functions::{activation::relu::relu, loss::cross_entropy::cross_entropy, softmax::softmax},
    Variable,
};
use zenu_layer::{layers::linear::Linear, Module, StateDict};
use zenu_matrix::device::{cpu::Cpu, Device};
use zenu_optimizer::sgd::SGD;

#[derive(Serialize, Deserialize)]
pub struct SimpleModel<D: Device> {
    pub linear_1: Linear<f32, D>,
    pub linear_2: Linear<f32, D>,
}

impl<'de, D: Device + Deserialize<'de>> StateDict<'de> for SimpleModel<D> {}

impl<D: Device> SimpleModel<D> {
    pub fn new() -> Self {
        Self {
            linear_1: Linear::new(28 * 28, 512, true),
            linear_2: Linear::new(512, 10, true),
        }
    }
}

impl<D: Device> Module<f32, D> for SimpleModel<D> {
    fn call(&self, inputs: zenu_autograd::Variable<f32, D>) -> Variable<f32, D> {
        let x = self.linear_1.call(inputs);
        let x = relu(x);
        let x = self.linear_2.call(x);
        softmax(x, 1)
    }
}

struct MnistDataset {
    data: Vec<(Vec<u8>, u8)>,
}

impl Dataset<f32> for MnistDataset {
    type Item = (Vec<u8>, u8);

    fn item(&self, item: usize) -> Vec<Variable<f32, Cpu>> {
        let (x, y) = &self.data[item];
        let x_f32 = x.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let x = from_vec::<f32, _, Cpu>(x_f32, [784]);
        let y_onehot = (0..10)
            .map(|i| if i == *y as usize { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let y = from_vec(y_onehot, [10]);
        vec![x, y]
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn all_data(&mut self) -> &mut [Self::Item] {
        &mut self.data as &mut [Self::Item]
    }
}

fn main() {
    let model = SimpleModel::<Cpu>::new();
    let (train, test) = mnist_dataset().unwrap();
    let (train, val) = train_val_split(&train, 0.8, true);

    let test_dataloader = DataLoader::new(MnistDataset { data: test }, 1);

    let optimizer = SGD::<f32, Cpu>::new(0.01);

    for num_epoch in 0..10 {
        let mut train_dataloader = DataLoader::new(
            MnistDataset {
                data: train.clone(),
            },
            32,
        );

        train_dataloader.shuffle();

        let mut train_loss = 0.0;
        let mut num_iter = 0;

        for batch in train_dataloader {
            let input = batch[0].clone();
            let target = batch[1].clone();
            let pred = model.call(input);
            let loss = cross_entropy(pred, target);
            update_parameters(loss.clone(), &optimizer);
            train_loss += loss.get_data().asum();
            num_iter += 1;
        }
        train_loss /= num_iter as f32;

        let val_loader = DataLoader::new(MnistDataset { data: val.clone() }, 16);

        let mut val_loss = 0.0;
        let mut num_iter = 0;
        for batch in val_loader {
            let input = batch[0].clone();
            let target = batch[1].clone();
            let pred = model.call(input);
            let loss = cross_entropy(pred, target);
            val_loss += loss.get_data().asum();
            num_iter += 1;
        }
        val_loss /= num_iter as f32;

        println!(
            "Epoch: {}, Train Loss: {}, Val Loss: {}",
            num_epoch, train_loss, val_loss
        );
    }

    let mut test_loss = 0.;
    let mut num_iter = 0;

    for batch in test_dataloader {
        let input = batch[0].clone();
        let target = batch[1].clone();
        let pred = model.call(input);
        let loss = cross_entropy(pred, target);
        test_loss += loss.get_data().asum();
        num_iter += 1;
    }

    println!("Test Loss: {}", test_loss / num_iter as f32);
}

use zenu::{
    autograd::{
        activation::relu::relu, creator::from_vec::from_vec, loss::cross_entropy::cross_entropy,
        no_train, set_train, Variable,
    },
    dataset::{train_val_split, DataLoader, Dataset},
    dataset_loader::mnist_dataset,
    layer::{layers::linear::Linear, Module},
    matrix::{
        device::{cpu::Cpu, Device},
        num::Num,
    },
    optimizer::{sgd::SGD, Optimizer},
};
use zenu_macros::Parameters;

#[derive(Parameters)]
#[parameters(num=T, device=D)]
pub struct SimpleModel<T: Num, D: Device> {
    pub linear_1: Linear<T, D>,
    pub linear_2: Linear<T, D>,
}

impl<D: Device> SimpleModel<f32, D> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            linear_1: Linear::new(28 * 28, 512, true),
            linear_2: Linear::new(512, 10, true),
        }
    }
}

impl<D: Device> Default for SimpleModel<f32, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Device> Module<f32, D> for SimpleModel<f32, D> {
    type Input = Variable<f32, D>;
    type Output = Variable<f32, D>;

    fn call(&self, inputs: Variable<f32, D>) -> Variable<f32, D> {
        let x = self.linear_1.call(inputs);
        let x = relu(x);
        self.linear_2.call(x)
    }
}

struct MnistDataset {
    data: Vec<(Vec<u8>, u8)>,
}

impl Dataset<f32> for MnistDataset {
    type Item = (Vec<u8>, u8);

    fn item(&self, item: usize) -> Vec<Variable<f32, Cpu>> {
        let (x, y) = &self.data[item];
        let x_f32 = x.iter().map(|&x| f32::from(x)).collect::<Vec<_>>();
        let x = from_vec::<f32, _, Cpu>(x_f32, [784]);
        x.get_data_mut().to_ref_mut().div_scalar_assign(127.5);
        x.get_data_mut().to_ref_mut().sub_scalar_assign(1.0);
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

#[expect(clippy::cast_precision_loss)]
fn main() {
    let model = SimpleModel::<f32, Cpu>::new();
    let (train, test) = mnist_dataset().unwrap();
    let (train, val) = train_val_split(&train, 0.8, true);

    let test_dataloader = DataLoader::new(MnistDataset { data: test }, 1);

    let optimizer = SGD::<f32, Cpu>::new(0.01);

    for num_epoch in 0..20 {
        set_train();
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
            let loss_asum = loss.get_data().asum();
            // update_parameters(&loss, &optimizer);
            loss.backward();
            optimizer.update(&model);
            loss.clear_grad();
            train_loss += loss_asum;
            num_iter += 1;
        }
        train_loss /= num_iter as f32;

        let val_loader = DataLoader::new(MnistDataset { data: val.clone() }, 1);

        no_train();
        let mut val_loss = 0.0;
        let mut num_iter = 0;
        for batch in val_loader {
            let input = batch[0].clone();
            let target = batch[1].clone();
            let pred = model.call(input);
            let loss = cross_entropy(pred.clone(), target.clone());
            val_loss += loss.get_data().asum();
            num_iter += 1;
        }
        val_loss /= num_iter as f32;

        println!("Epoch: {num_epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}");
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

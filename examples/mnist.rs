use zenu::{
    dataset::{train_val_split, DataLoader, Dataset},
    mnist::minist_dataset,
    update_parameters, Model,
};
use zenu_autograd::{
    creator::from_vec::from_vec,
    functions::{activation::sigmoid::sigmoid, loss::cross_entropy::cross_entropy},
    Variable,
};
use zenu_layer::{layers::linear::Linear, Layer};
use zenu_matrix::matrix::IndexItem;
use zenu_optimizer::sgd::SGD;

struct SingleLayerModel {
    linear: Linear<f32>,
}

impl SingleLayerModel {
    fn new() -> Self {
        let mut linear = Linear::new(784, 10);
        linear.init_parameters(None);
        Self { linear }
    }
}

impl Model<f32> for SingleLayerModel {
    fn predict(&self, inputs: &[Variable<f32>]) -> Variable<f32> {
        let x = &inputs[0];
        let x = self.linear.call(x.clone());
        sigmoid(x)
    }
}

struct MnistDataset {
    data: Vec<(Vec<u8>, u8)>,
}

impl Dataset<f32> for MnistDataset {
    type Item = (Vec<u8>, u8);

    fn item(&self, item: usize) -> Vec<Variable<f32>> {
        let (x, y) = &self.data[item];
        let x_f32 = x.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let x = from_vec(x_f32, [784]);
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
    let (train, test) = minist_dataset().unwrap();
    let (train, val) = train_val_split(&train, 0.8, true);

    let test_dataloader = DataLoader::new(MnistDataset { data: test }, 16);

    let sgd = SGD::new(0.01);
    let model = SingleLayerModel::new();

    for epoch in 0..10 {
        let mut train_dataloader = DataLoader::new(
            MnistDataset {
                data: train.clone(),
            },
            16,
        );
        let val_dataloader = DataLoader::new(MnistDataset { data: val.clone() }, 16);

        train_dataloader.shuffle();

        let mut epoch_loss_train: f32 = 0.;
        let mut num_iter_train = 0;
        for batch in train_dataloader {
            let input = batch[0].clone();
            let target = batch[1].clone();
            let y_pred = model.predict(&[input]);
            let loss = cross_entropy(y_pred, target);
            update_parameters(loss.clone(), &sgd);
            epoch_loss_train += loss.get_data().index_item([]);
            num_iter_train += 1;
        }

        let mut epoch_loss_val = 0.;
        let mut num_iter_val = 0;
        for batch in val_dataloader {
            let input = batch[0].clone();
            let target = batch[1].clone();
            let y_pred = model.predict(&[input]);
            let loss = cross_entropy(y_pred, target);
            epoch_loss_val += loss.get_data().index_item([]);
            num_iter_val += 1;
        }

        println!(
            "Epoch: {}, Train Loss: {}, Val Loss: {}",
            epoch,
            epoch_loss_train / num_iter_train as f32,
            epoch_loss_val / num_iter_val as f32
        );
    }

    let mut test_loss = 0.;
    let mut num_iter_test = 0;
    for batch in test_dataloader {
        let input = batch[0].clone();
        let target = batch[1].clone();
        let y_pred = model.predict(&[input]);
        let loss = cross_entropy(y_pred, target);
        test_loss += loss.get_data().index_item([]);
        num_iter_test += 1;
    }

    println!("Test Loss: {}", test_loss / num_iter_test as f32);
}

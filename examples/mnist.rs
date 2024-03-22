use zenu::{dataset::train_val_split, mnist::minist_dataset, update_parameters, Model};
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

fn main() {
    let (train, test) = minist_dataset().unwrap();
    let (train, val) = train_val_split(&train, 0.8, true);
    let sgd = SGD::new(0.01);
    let model = SingleLayerModel::new();

    for epoch in 0..10 {
        let mut epoch_loss_train: f32 = 0.;
        let mut num_iter_train = 0;
        for (x, y) in train.iter() {
            let x_f32 = x.iter().map(|&x| x as f32).collect::<Vec<_>>();
            let y_onehot = (0..10)
                .map(|i| if i == *y as usize { 1.0 } else { 0.0 })
                .collect::<Vec<_>>();
            let x = from_vec(x_f32, [1, 784]);
            let y = from_vec(y_onehot, [1, 10]);
            let y_pred = model.predict(&[x]);
            let loss = cross_entropy(y_pred, y);
            update_parameters(loss.clone(), &sgd);
            epoch_loss_train += loss.get_data().index_item([]);
            num_iter_train += 1;
            if num_iter_train % 100 == 0 {
                println!(
                    "Epoch: {}, Iter: {}, Loss: {}",
                    epoch,
                    num_iter_train,
                    epoch_loss_train / num_iter_train as f32
                );
            }
        }
        let mut epoch_loss_val = 0.;
        let mut num_iter_val = 0;
        for (x, y) in val.iter() {
            let x_f32 = x.iter().map(|&x| x as f32).collect::<Vec<_>>();
            let y_onehot = (0..10)
                .map(|i| if i == *y as usize { 1.0 } else { 0.0 })
                .collect::<Vec<_>>();
            let x = from_vec(x_f32, [1, 784]);
            let y = from_vec(y_onehot, [1, 10]);
            let y_pred = model.predict(&[x]);
            let loss = cross_entropy(y_pred, y);
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

    for (x, y) in test.iter() {
        let x_f32 = x.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let y_onehot = (0..10)
            .map(|i| if i == *y as usize { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let x = from_vec(x_f32, [1, 784]);
        let y = from_vec(y_onehot, [1, 10]);
        let y_pred = model.predict(&[x]);
        let loss = cross_entropy(y_pred, y);
        test_loss += loss.get_data().index_item([]);
        num_iter_test += 1;
    }

    println!("Test Loss: {}", test_loss / num_iter_test as f32);
}

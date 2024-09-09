use rand::seq::SliceRandom;

use zenu_autograd::{concat::concat, Variable};
use zenu_matrix::{device::cpu::Cpu, num::Num};

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn train_val_split<T: Clone>(data: &[T], split_ratio: f64, shuffle: bool) -> (Vec<T>, Vec<T>) {
    let mut data = data.to_vec();
    if shuffle {
        let mut rng = rand::thread_rng();
        data.shuffle(&mut rng);
    }
    let split_idx = (data.len() as f64 * split_ratio).round() as usize;
    let (train, val) = data.split_at(split_idx);
    (train.to_vec(), val.to_vec())
}

pub trait Dataset<T: Num> {
    type Item;
    fn item(&self, index: usize) -> Vec<Variable<T, Cpu>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn all_data(&mut self) -> &mut [Self::Item];
    fn shuffle(&mut self) {
        let mut rng = rand::thread_rng();
        self.all_data().shuffle(&mut rng);
    }
}

pub struct DataLoader<T: Num, D: Dataset<T>> {
    dataset: D,
    batch_size: usize,
    index: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Num, D: Dataset<T>> DataLoader<T, D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        DataLoader {
            dataset,
            batch_size,
            index: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    pub fn len(&self) -> usize {
        (self.dataset.len() as f64 / self.batch_size as f64).ceil() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.len() == 0
    }

    pub fn shuffle(&mut self) {
        self.dataset.shuffle();
    }
}

impl<T: Num, D: Dataset<T>> Iterator for DataLoader<T, D> {
    type Item = Vec<Variable<T, Cpu>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index * self.batch_size >= self.dataset.len() {
            return None;
        }
        let end_idx = std::cmp::min((self.index + 1) * self.batch_size, self.dataset.len());
        let batch: Vec<Vec<Variable<T, Cpu>>> = (self.index * self.batch_size..end_idx)
            .map(|i| self.dataset.item(i))
            .collect();

        self.index += 1;

        let k = batch[0].len();
        for v in batch.iter().skip(1) {
            assert_eq!(v.len(), k, "All dataset's output size must be same");
        }

        let mut result = vec![vec![]; k];

        for item in batch {
            for (i, v) in item.iter().enumerate() {
                result[i].push(v.clone());
            }
        }

        let result: Vec<Variable<T, Cpu>> = result.iter().map(|v| concat(v)).collect();

        if result.len() == 1 {
            Some(result)
        } else {
            let first_batch_size = result[0].get_data().shape()[0];
            for v in result.iter().skip(1) {
                assert_eq!(v.get_data().shape()[0], first_batch_size);
            }
            Some(result)
        }
    }
}

#[cfg(test)]
mod dataset_tests {
    use zenu_autograd::{creator::from_vec::from_vec, Variable};
    use zenu_matrix::{
        device::cpu::Cpu,
        dim::{DimDyn, DimTrait},
        matrix::{Matrix, Owned},
    };

    use super::{DataLoader, Dataset};

    struct DummyDataset {
        data: Vec<Vec<f64>>,
    }

    impl DummyDataset {
        fn new(data: Vec<Vec<f64>>) -> Self {
            DummyDataset { data }
        }
    }

    impl Dataset<f64> for DummyDataset {
        type Item = Vec<f64>;

        fn item(&self, index: usize) -> Vec<Variable<f64, Cpu>> {
            vec![from_vec(self.data[index].clone(), [self.data[index].len()])]
        }

        fn len(&self) -> usize {
            self.data.len()
        }

        fn all_data(&mut self) -> &mut [Self::Item] {
            &mut self.data
        }
    }

    struct DummyDataset2 {
        data: Vec<(Vec<f64>, u8)>,
    }

    impl DummyDataset2 {
        fn new(data: Vec<(Vec<f64>, u8)>) -> Self {
            DummyDataset2 { data }
        }
    }

    impl Dataset<f64> for DummyDataset2 {
        type Item = (Vec<f64>, u8);

        fn item(&self, index: usize) -> Vec<Variable<f64, Cpu>> {
            let first_elm = from_vec(self.data[index].0.clone(), [self.data[index].0.len()]);
            // onehot
            let mut v = [0.; 10];
            v[self.data[index].1 as usize] = 1.;
            let second_elm = from_vec(v.to_vec(), [10]);
            vec![first_elm, second_elm]
        }

        fn len(&self) -> usize {
            self.data.len()
        }

        fn all_data(&mut self) -> &mut [(Vec<f64>, u8)] {
            &mut self.data
        }
    }

    #[test]
    fn dummy_dataset_1() {
        let data = vec![
            vec![1., 2., 3.],
            vec![4., 5., 6.],
            vec![7., 8., 9.],
            vec![10., 11., 12.],
            vec![13., 14., 15.],
            vec![16., 17., 18.],
            vec![19., 20., 21.],
        ];

        let dataset = DummyDataset::new(data);
        let mut dataloader = DataLoader::new(dataset, 2);

        let batch = &dataloader.next().unwrap()[0];
        let expected_batch =
            Matrix::<Owned<f64>, DimDyn, Cpu>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let diff = batch.get_data().to_ref() - expected_batch;
        assert_eq!(diff.asum(), 0.);

        let batch = &dataloader.next().unwrap()[0];
        let expected_batch =
            Matrix::<Owned<f64>, DimDyn, Cpu>::from_vec(vec![7., 8., 9., 10., 11., 12.], [2, 3]);
        let diff = batch.get_data().to_ref() - expected_batch;
        assert_eq!(diff.asum(), 0.);

        let batch = &dataloader.next().unwrap()[0];
        let expected_batch =
            Matrix::<Owned<f64>, DimDyn, Cpu>::from_vec(vec![13., 14., 15., 16., 17., 18.], [2, 3]);
        let diff = batch.get_data().to_ref() - expected_batch;
        assert_eq!(diff.asum(), 0.);

        let batch = &dataloader.next().unwrap()[0];
        let expected_batch =
            Matrix::<Owned<f64>, DimDyn, Cpu>::from_vec(vec![19., 20., 21.], [1, 3]);
        let diff = batch.get_data().to_ref() - expected_batch;
        assert_eq!(diff.asum(), 0.);
    }

    #[test]
    fn dummy_dataset_2() {
        let data = vec![
            (vec![1., 2., 3.], 0),
            (vec![4., 5., 6.], 1),
            (vec![7., 8., 9.], 2),
            (vec![10., 11., 12.], 3),
            (vec![13., 14., 15.], 4),
            (vec![16., 17., 18.], 5),
            (vec![19., 20., 21.], 6),
        ];

        let dataset = DummyDataset2::new(data);
        let mut dataloader = DataLoader::new(dataset, 2);

        let batch = &dataloader.next().unwrap();
        let expected_batch =
            Matrix::<Owned<f64>, DimDyn, Cpu>::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let diff = batch[0].get_data().to_ref() - expected_batch;
        assert_eq!(diff.asum(), 0.);
        assert_eq!(batch[1].get_data().to_ref().shape().slice(), [2, 10]);
        let expected_batch = Matrix::<Owned<f64>, DimDyn, Cpu>::from_vec(
            vec![
                1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            [2, 10],
        );
        let diff = batch[1].get_data().to_ref() - expected_batch;
        assert_eq!(diff.asum(), 0.);
    }
}

use rand::seq::SliceRandom;

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

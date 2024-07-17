use std::env;
use std::fs;
use std::fs::File;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::{Read, Write};
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use flate2::read::GzDecoder;
use reqwest::blocking::get;
use tar::Archive;

enum Dataset {
    Mnist,
    Cifar10,
}

const MNIST_URLS: [&str; 4] = [
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
];

const MNIST_FILENAMES: [&str; 4] = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
];

const CIFAR10_URLS: [&str; 1] = ["https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"];

const CIFAR10_FILENAMES: [&str; 1] = ["cifar-10-binary.tar.gz"];

#[allow(clippy::type_complexity)]
pub fn mnist_dataset(
) -> Result<(Vec<(Vec<u8>, u8)>, Vec<(Vec<u8>, u8)>), Box<dyn std::error::Error>> {
    let dataset_dir = create_dataset_dir("mnist")?;
    download_dataset(&Dataset::Mnist, &dataset_dir)?;
    let (train_data, test_data) = extract_image_label_pairs(&Dataset::Mnist, &dataset_dir)?;
    Ok((train_data, test_data))
}

#[allow(clippy::type_complexity)]
pub fn cifar10_dataset(
) -> Result<(Vec<(Vec<u8>, u8)>, Vec<(Vec<u8>, u8)>), Box<dyn std::error::Error>> {
    let dataset_dir = create_dataset_dir("cifar10")?;
    download_and_extract_cifar10(dataset_dir.to_str().unwrap())?;
    let dataset_dir = dataset_dir.join("cifar-10-batches-py");
    let (train_data, _) = extract_image_label_pairs(&Dataset::Cifar10, &dataset_dir)?;
    Ok((train_data, Vec::new()))
}

fn download_and_extract_cifar10(save_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz";
    let filename = "cifar-10-python.tar.gz";
    let save_path = Path::new(save_dir).join(filename);

    if save_path.exists() {
        println!("ファイルは既に存在します: {}", save_path.display());
    } else {
        if !Path::new(save_dir).exists() {
            fs::create_dir_all(save_dir)?;
        }

        let output = Command::new("wget")
            .arg("-O") // 出力ファイル名を指定
            .arg(&save_path)
            .arg(url)
            .output()?;

        if output.status.success() {
            println!(
                "ファイルのダウンロードが完了しました: {}",
                save_path.display()
            );
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(format!("ダウンロード中にエラーが発生しました: {}", error).into());
        }
    }

    let output = Command::new("tar")
        .arg("-xzf")
        .arg(&save_path)
        .arg("-C")
        .arg(save_dir)
        .output()?;

    if output.status.success() {
        println!("ファイルの解凍が完了しました");
        Ok(())
    } else {
        let error = String::from_utf8_lossy(&output.stderr);
        Err(format!("解凍中にエラーが発生しました: {}", error).into())
    }
}

fn download_dataset(
    dataset: &Dataset,
    dataset_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let (urls, filenames) = match dataset {
        Dataset::Mnist => (MNIST_URLS.as_ref(), MNIST_FILENAMES.as_ref()),
        Dataset::Cifar10 => (CIFAR10_URLS.as_ref(), CIFAR10_FILENAMES.as_ref()),
    };

    for (url, filename) in urls.iter().zip(filenames.iter()) {
        let filepath = dataset_dir.join(filename);

        if !filepath.exists() {
            println!("Downloading: {}", url);
            let response = get(*url)?;
            println!("response is {:?}", response.status());
            let mut file = File::create(&filepath)?;
            let content = response.bytes()?;
            file.write_all(&content)?;
            println!("Downloaded: {:?}", filepath);

            println!("filename: {:?}", filename);
            // 解凍処理を追加
            if filename.ends_with(".tar.gz") {
                println!("Extracting: {:?}", filepath);
                let tar_gz = File::open(&filepath)?;
                let tar = GzDecoder::new(tar_gz);
                let mut archive = Archive::new(tar);
                archive.unpack(dataset_dir)?;
                println!("Extracted: {:?}", dataset_dir);
            } else if filename.ends_with(".gz") {
                let unzipped_filename = filename.replace(".gz", "");
                let unzipped_filepath = dataset_dir.join(&unzipped_filename);
                println!("Unzipping: {:?}", filepath);
                let mut gz = GzDecoder::new(File::open(&filepath)?);
                let mut unzipped_file = File::create(&unzipped_filepath)?;
                std::io::copy(&mut gz, &mut unzipped_file)?;
                println!("Unzipped: {:?}", unzipped_filepath);
            }
        } else {
            println!("File already exists: {:?}", filepath);
        }
    }

    Ok(())
}

fn create_dataset_dir(dataset_name: &str) -> Result<PathBuf, std::io::Error> {
    let home_dir = if cfg!(target_os = "windows") {
        env::var("USERPROFILE").expect("Failed to get home directory")
    } else {
        env::var("HOME").expect("Failed to get home directory")
    };

    let target_dir = PathBuf::from(home_dir).join(format!(".zenu/data/{}", dataset_name));

    if !target_dir.exists() {
        fs::create_dir_all(&target_dir)?;
        println!("Directory created: {:?}", target_dir);
    } else {
        println!("Directory already exists: {:?}", target_dir);
    }

    Ok(target_dir)
}

#[allow(clippy::type_complexity)]
fn extract_image_label_pairs(
    dataset: &Dataset,
    dataset_dir: &Path,
) -> Result<(Vec<(Vec<u8>, u8)>, Vec<(Vec<u8>, u8)>), Box<dyn std::error::Error>> {
    println!("dataset_dir: {:?}", dataset_dir);
    match dataset {
        Dataset::Mnist => extract_mnist_image_label_pairs(dataset_dir),
        Dataset::Cifar10 => {
            let train_data = extract_cifar10_image_label_pairs(dataset_dir)?;
            Ok((train_data, Vec::new()))
        }
    }
}

#[allow(clippy::type_complexity)]
fn extract_mnist_image_label_pairs(
    dataset_dir: &Path,
) -> Result<(Vec<(Vec<u8>, u8)>, Vec<(Vec<u8>, u8)>), Box<dyn std::error::Error>> {
    let train_images_path = dataset_dir.join("train-images-idx3-ubyte");
    let train_labels_path = dataset_dir.join("train-labels-idx1-ubyte");
    let test_images_path = dataset_dir.join("t10k-images-idx3-ubyte");
    let test_labels_path = dataset_dir.join("t10k-labels-idx1-ubyte");

    let mut train_images_file = File::open(train_images_path)?;
    let mut train_labels_file = File::open(train_labels_path)?;
    let mut test_images_file = File::open(test_images_path)?;
    let mut test_labels_file = File::open(test_labels_path)?;

    // 画像ファイルのヘッダーを読み込む
    let mut train_images_header = [0; 16];
    train_images_file.read_exact(&mut train_images_header)?;
    let num_train_images = u32::from_be_bytes([
        train_images_header[4],
        train_images_header[5],
        train_images_header[6],
        train_images_header[7],
    ]) as usize;

    let mut test_images_header = [0; 16];
    test_images_file.read_exact(&mut test_images_header)?;
    let num_test_images = u32::from_be_bytes([
        test_images_header[4],
        test_images_header[5],
        test_images_header[6],
        test_images_header[7],
    ]) as usize;

    // ラベルファイルのヘッダーを読み飛ばす
    train_labels_file.seek(SeekFrom::Start(8))?;
    test_labels_file.seek(SeekFrom::Start(8))?;

    // 訓練データとテストデータの画像とラベルのペアを抽出する
    let mut train_data = Vec::with_capacity(num_train_images);
    let mut test_data = Vec::with_capacity(num_test_images);

    let mut image_buffer = vec![0; 784];
    let mut label_buffer = [0; 1];

    for _ in 0..num_train_images {
        train_images_file.read_exact(&mut image_buffer)?;
        train_labels_file.read_exact(&mut label_buffer)?;
        train_data.push((image_buffer.clone(), label_buffer[0]));
    }

    for _ in 0..num_test_images {
        test_images_file.read_exact(&mut image_buffer)?;
        test_labels_file.read_exact(&mut label_buffer)?;
        test_data.push((image_buffer.clone(), label_buffer[0]));
    }

    Ok((train_data, test_data))
}

#[allow(clippy::type_complexity)]
fn extract_cifar10_image_label_pairs(
    dataset_dir: &Path,
) -> Result<Vec<(Vec<u8>, u8)>, Box<dyn std::error::Error>> {
    let mut data = Vec::new();

    for i in 1..=5 {
        let file_path = dataset_dir.join(format!("data_batch_{}", i));
        let mut file = File::open(file_path)?;
        let mut buffer = [0; 3073];

        while file.read_exact(&mut buffer).is_ok() {
            let label = buffer[0];
            let image = buffer[1..].to_vec();
            data.push((image, label));
        }
    }

    Ok(data)
}

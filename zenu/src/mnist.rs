use std::env;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::path::PathBuf;

use flate2::read::GzDecoder;
use reqwest::blocking::get;

pub fn minist_dataset(
) -> Result<(Vec<(Vec<u8>, u8)>, Vec<(Vec<u8>, u8)>), Box<dyn std::error::Error>> {
    let dataset_dir = create_minist_dir()?;
    download_mnist_dataset(&dataset_dir)?;
    let (train_data, test_data) = extract_image_label_pairs(&dataset_dir)?;
    Ok((train_data, test_data))
}

fn create_minist_dir() -> Result<PathBuf, std::io::Error> {
    let home_dir = if cfg!(target_os = "windows") {
        env::var("USERPROFILE").expect("Failed to get home directory")
    } else {
        env::var("HOME").expect("Failed to get home directory")
    };

    let target_dir = PathBuf::from(home_dir).join(".zenu/data/minist");

    if !target_dir.exists() {
        fs::create_dir_all(&target_dir)?;
        println!("Directory created: {:?}", target_dir);
    } else {
        println!("Directory already exists: {:?}", target_dir);
    }

    Ok(target_dir)
}

fn download_mnist_dataset(dataset_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let mnist_urls = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ];

    for url in mnist_urls {
        let filename = url.split('/').last().unwrap();
        let filepath = dataset_dir.join(filename);

        if !filepath.exists() {
            println!("Downloading: {}", url);
            let response = get(url)?;
            let mut file = File::create(&filepath)?;
            let content = response.bytes()?;
            file.write_all(&content)?;
            println!("Downloaded: {:?}", filepath);

            // 解凍処理を追加
            let unzipped_filename = filename.replace(".gz", "");
            let unzipped_filepath = dataset_dir.join(unzipped_filename);
            println!("Unzipping: {:?}", filepath);
            let mut gz = GzDecoder::new(File::open(&filepath)?);
            let mut unzipped_file = File::create(&unzipped_filepath)?;
            std::io::copy(&mut gz, &mut unzipped_file)?;
            println!("Unzipped: {:?}", unzipped_filepath);
        } else {
            println!("File already exists: {:?}", filepath);
        }
    }

    Ok(())
}

fn extract_image_label_pairs(
    dataset_dir: &PathBuf,
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

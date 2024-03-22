use std::env;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use flate2::read::GzDecoder;
use reqwest::blocking::get;
use zenu_autograd::Variable;
use zenu_matrix::num::Num;
use zenu_optimizer::Optimizer;

pub trait Model<T: Num> {
    fn predict(&self, inputs: &[Variable<T>]) -> Variable<T>;
}

pub fn update<T: Num, O: Optimizer<T>>(loss: Variable<T>, optimizer: O) {
    loss.backward();
    let parameters = loss.get_all_trainable_variables();
    optimizer.update(&parameters);
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

pub fn download_mnist_dataset() -> Result<(), Box<dyn std::error::Error>> {
    let dataset_dir = create_minist_dir()?;
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

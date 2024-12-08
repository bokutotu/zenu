extern crate bindgen;
extern crate cmake;

use std::env;
use std::path::PathBuf;

fn main() {
    // 環境変数OUT_DIRを取得 (cmake::Configでbuild()すると自動的にこのフォルダにビルド成果物が出力される)
    let out_dir = env::var("OUT_DIR").unwrap();

    // CMakeによるビルド
    let dst = cmake::Config::new("cudnn_frontend_wrapper")
        .define("CMAKE_INSTALL_PREFIX", &out_dir)
        .build();

    // リンク検索パスを追加（libcudnn_frontend_wrapper.aが存在する場所）
    println!("cargo:rustc-link-search=native={}/lib", dst.display());

    // 静的ライブラリのリンク指定
    println!("cargo:rustc-link-lib=static=cudnn_frontend_wrapper");

    // 必要に応じてCUDA、cuDNNのリンクを指定（ライブラリが静的にすべてリンク済みなら不要）
    // 例: println!("cargo:rustc-link-lib=cudart");
    //     println!("cargo:rustc-link-lib=cudnn");

    // BindgenでヘッダファイルからRust用バインディングを生成
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let header_path = PathBuf::from(&manifest_dir)
        .join("cudnn_frontend_wrapper")
        .join("include")
        .join("cudnn_frontend_wrapper.h");

    let bindings = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .clang_arg(format!(
            "-I{}",
            PathBuf::from(&manifest_dir)
                .join("cudnn_frontend_wrapper/include")
                .display()
        ))
        .clang_arg("-I/usr/local/cuda/include")
        .generate()
        .expect("Unable to generate bindings");

    // srcディレクトリにbindings.rsとして出力
    let out_path = PathBuf::from("src").join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("Couldn't write bindings!");
}

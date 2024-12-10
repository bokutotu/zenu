extern crate bindgen;
extern crate cmake;

use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let dst = cmake::Config::new("cudnn_frontend_wrapper")
        .define("CMAKE_INSTALL_PREFIX", &out_dir)
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());

    println!("cargo:rustc-link-lib=static=cudnn_frontend_wrapper");

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
        .rustified_enum(".*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from("src").join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("Couldn't write bindings!");
}

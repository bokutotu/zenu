extern crate bindgen;

use std::env;
use std::path::PathBuf;
use zenu_cuda_config::find_cuda;

fn main() {
    for path in find_cuda() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rerun-if-changed=build.rs");
    let bindings = bindgen::Builder::default()
        .ctypes_prefix("::libc")
        .allowlist_function("cu.*")
        .allowlist_var("CUDNN.*")
        .allowlist_type("[Cc][Uu].*")
        .default_alias_style(bindgen::AliasVariation::TypeAlias)
        .rustified_non_exhaustive_enum("cudnn[A-Za-z]+_t")
        .rustified_non_exhaustive_enum("cuda.*")
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        // .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // .rustfmt_bindings(true)
        .clang_arg("-I")
        .clang_arg("/usr/local/cuda/include".to_string())
        .header("wrapper.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Unable to write");
}

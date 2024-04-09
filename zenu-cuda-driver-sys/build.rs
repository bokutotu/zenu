extern crate bindgen;

use std::{env, path::PathBuf};
use zenu_cuda_config::find_cuda;

fn main() {
    for path in find_cuda() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rerun-if-changed=build.rs");

    let bindings = bindgen::Builder::default()
        .ctypes_prefix("::libc")
        .size_t_is_usize(true)
        .clang_arg("-I")
        .clang_arg("/usr/local/cuda/include".to_string())
        .header("wrapper.h")
        .rustified_non_exhaustive_enum("cuda.*")
        .allowlist_type("^cuda.*")
        .allowlist_type("^surfaceReference")
        .allowlist_type("^textureReference")
        .allowlist_var("^cuda.*")
        .allowlist_function("^cuda.*")
        .default_alias_style(bindgen::AliasVariation::TypeAlias)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        // .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // .rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Unable to write");
}

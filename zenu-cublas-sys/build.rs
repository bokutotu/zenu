extern crate bindgen;

use zenu_cuda_config::find_cuda;

fn main() {
    for path in find_cuda() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-lib=dylib=cublas");

    let bindings = bindgen::Builder::default()
        .ctypes_prefix("::libc")
        .size_t_is_usize(true)
        .clang_arg("-I")
        .clang_arg("/usr/local/cuda/include".to_string())
        .header("wrapper.h")
        .rustified_non_exhaustive_enum("cublas[A-Za-z]+_t")
        // .rustified_non_exhaustive_enum("cuda.*")
        .default_alias_style(bindgen::AliasVariation::TypeAlias)
        // .rustfmt_bindings(true)
        .allowlist_type("^cublas.*")
        .allowlist_type("cublas.*")
        .allowlist_function("^cublas.*")
        .allowlist_type("[Cc][Uu].*")
        .allowlist_var("CUBLAS.*")
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        // .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("./src/bindings.rs")
        .expect("Unable to write");
}

extern crate bindgen;
extern crate cc;

fn main() {
    let cuda_files = vec![
        "kernel/array_scalar.cu",
        "kernel/element_wise.cu",
        "kernel/memory_access.cu",
        "kernel/array_array.cu",
        "kernel/activations.cu",
        "kernel/conv2d_bkwd_data.cu",
    ];

    for cuda_file in &cuda_files {
        println!("cargo:rerun-if-changed={cuda_file}");
    }
    println!("cargo:rerun-if-changed=kernel/kernel.h");
    println!("cargo:rerun-if-changed=build.rs");

    cc::Build::new()
        .cuda(true)
        .cpp(true)
        .flag("-std=c++11")
        .flag("-cudart=shared")
        .flag("--expt-relaxed-constexpr")
        .files(cuda_files)
        .include("kernel/")
        .compile("libkernel.a");

    println!("cargo:rustc-link-lib=kernel");
    println!("cargo:rustc-link-lib=static=kernel");
    println!(
        "cargo:rustc-link-search=native={}",
        std::env::var("OUT_DIR").unwrap()
    );

    let bindings = bindgen::Builder::default()
        .ctypes_prefix("::libc")
        .size_t_is_usize(true)
        .clang_arg("-I")
        .clang_arg("/usr/local/cuda/include".to_string())
        .header("kernel/kernel.h")
        .default_alias_style(bindgen::AliasVariation::TypeAlias)
        .derive_default(true)
        .derive_eq(true)
        .derive_hash(true)
        .derive_ord(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("src/bindings.rs")
        .expect("Couldn't write bindings!");
}

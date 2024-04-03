#[macro_use]
extern crate criterion;

use criterion::black_box;
use criterion::Criterion;
use zenu_autograd::creator::ones::ones;
use zenu_autograd::functions::conv2d::conv2d;
use zenu_autograd::Variable;

fn conv2d_bench(kernel: Variable<f32>, input: Variable<f32>) {
    let _ = conv2d(input, kernel, None, (1, 1), (0, 0));
}

fn conv2d_bench_no_bias(c: &mut Criterion) {
    let kernel = black_box(ones([32, 16, 3, 3]));
    let input = black_box(ones([32, 16, 128, 128]));

    c.bench_function("conv2d_bech_no_bias", |b| {
        b.iter(|| conv2d_bench(kernel.clone(), input.clone()))
    });
}

criterion_group!(benches, conv2d_bench_no_bias);
criterion_main!(benches);

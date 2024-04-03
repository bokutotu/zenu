use criterion::Criterion;
use zenu_matrix::{
    dim::{DimDyn, DimTrait},
    shape_stride::ShapeStride,
};

#[inline]
fn generate_combinations(dimensions: &[usize]) -> Vec<Vec<usize>> {
    let total_combinations = dimensions.iter().product();
    let mut combinations = Vec::with_capacity(total_combinations);
    let mut current_combination = vec![0; dimensions.len()];

    for _ in 0..total_combinations {
        combinations.push(current_combination.clone());

        for i in (0..dimensions.len()).rev() {
            current_combination[i] += 1;
            if current_combination[i] < dimensions[i] {
                break;
            }
            current_combination[i] = 0;
        }
    }

    combinations
}

#[inline]
fn copy_apply_index(shape: DimDyn) -> Vec<DimDyn> {
    let slice = shape.slice();
    let combinations = generate_combinations(&slice);
    combinations
        .iter()
        .map(|combination| DimDyn::from(&combination as &[usize]))
        .collect()
}

#[inline]
fn get_max_shape_idx_of_apply_blas(a: ShapeStride<DimDyn>, b: ShapeStride<DimDyn>) -> usize {
    let min_len = std::cmp::min(a.shape().len(), b.shape().len());
    let a_len = a.shape().len();
    let b_len = b.shape().len();

    match min_len {
        0 => 0,
        1 => 1,
        2 => {
            let a_stride = a.stride();
            let b_stride = b.stride();
            let a_shape = a.shape();
            let b_shape = b.shape();
            let a_stride_part = DimDyn::from(&a_stride.slice()[a_len - 2..]);
            let b_stride_part = DimDyn::from(&b_stride.slice()[b_len - 2..]);
            let a_shape_part = DimDyn::from(&a_shape.slice()[a_len - 2..]);
            let b_shape_part = DimDyn::from(&b_shape.slice()[b_len - 2..]);
            let a_part = ShapeStride::new(a_shape_part, a_stride_part);
            let b_part = ShapeStride::new(b_shape_part, b_stride_part);
            if !(a_part.is_transposed() || b_part.is_transposed())
                && a_part.is_contiguous()
                && b_part.is_contiguous()
            {
                2
            } else {
                1
            }
        }
        _ => {
            let mut idx = 1;
            for i in 2..=min_len {
                let a_shape_part = DimDyn::from(&a.shape().slice()[a_len - i..]);
                let b_shape_part = DimDyn::from(&b.shape().slice()[b_len - i..]);
                let a_stride_part = DimDyn::from(&a.stride().slice()[a_len - i..]);
                let b_stride_part = DimDyn::from(&b.stride().slice()[b_len - i..]);
                let a_part = ShapeStride::new(a_shape_part, a_stride_part);
                let b_part = ShapeStride::new(b_shape_part, b_stride_part);
                if !a_part.is_transposed()
                    && (a_part.is_transposed() == b_part.is_transposed())
                    && a_part.is_contiguous()
                    && b_part.is_contiguous()
                {
                    idx = i;
                } else {
                    break;
                }
            }
            idx
        }
    }
}

#[inline]
fn combine_vecs(vec1: &[usize], vec2: &[usize]) -> Vec<(usize, usize)> {
    let len1 = vec1.len();
    let len2 = vec2.len();
    let max_len = len1.max(len2);

    let mut combined_vec = Vec::with_capacity(max_len);

    let mut iter1 = vec1.iter().cycle();
    let mut iter2 = vec2.iter().cycle();

    combined_vec.extend((0..max_len).map(|_| (*iter1.next().unwrap(), *iter2.next().unwrap())));

    combined_vec
}

fn copy_apply_index_bench(c: &mut Criterion) {
    let a = DimDyn::from(&[32, 16, 3, 3, 126]);
    let b = DimDyn::from(&[32, 16, 126, 126]);

    c.bench_function("copy_from_logic_copy_apply_index_bench", |b_| {
        b_.iter(|| {
            copy_apply_index(a);
            copy_apply_index(b)
        })
    });
}

#[inline]
fn get_all_blas_opset_stride(
    a: ShapeStride<DimDyn>,
    b: ShapeStride<DimDyn>,
) -> Vec<(usize, usize)> {
    let max_idx = get_max_shape_idx_of_apply_blas(a, b);
    let a_part_stride = DimDyn::from(&a.stride().slice()[..a.shape().len() - max_idx]);
    let b_part_stride = DimDyn::from(&b.stride().slice()[..b.shape().len() - max_idx]);
    let a_part_shape = DimDyn::from(&a.shape().slice()[..a.shape().len() - max_idx]);
    let b_part_shape = DimDyn::from(&b.shape().slice()[..b.shape().len() - max_idx]);

    let a_indexes = copy_apply_index(a_part_shape);
    let b_indexes = copy_apply_index(b_part_shape);

    let a_stride_offset = a_indexes
        .iter()
        .map(|index| {
            index
                .slice()
                .iter()
                .zip(a_part_stride.slice().iter())
                .fold(0, |acc, (&i, &s)| acc + i * s)
        })
        .collect::<Vec<_>>();

    let b_stride_offset = b_indexes
        .iter()
        .map(|index| {
            index
                .slice()
                .iter()
                .zip(b_part_stride.slice().iter())
                .fold(0, |acc, (&i, &s)| acc + i * s)
        })
        .collect::<Vec<_>>();

    combine_vecs(&a_stride_offset, &b_stride_offset)
}

fn bench_all(a: ShapeStride<DimDyn>, b: ShapeStride<DimDyn>) {
    let _ = get_all_blas_opset_stride(a, b);
}

fn max_index_bench(c: &mut Criterion) {
    let a = ShapeStride::new(
        DimDyn::from(&[32, 16, 3, 3, 126, 126]),
        DimDyn::from(&[
            126 * 126 * 3 * 3 * 16,
            126 * 126 * 3 * 3,
            126 * 126 * 3,
            126 * 126,
            126,
            1,
        ]),
    );
    let b = ShapeStride::new(
        DimDyn::from(&[32, 16, 126, 126]),
        DimDyn::from(&[128 * 128 * 16, 128 * 128, 128, 1]),
    );

    c.bench_function("copy_from_logic_max_index_bench", |b_| {
        b_.iter(|| get_max_shape_idx_of_apply_blas(a, b))
    });
}

fn bench(c: &mut Criterion) {
    let a = ShapeStride::new(
        DimDyn::from(&[32, 16, 3, 3, 126, 126]),
        DimDyn::from(&[
            126 * 126 * 3 * 3 * 16,
            126 * 126 * 3 * 3,
            126 * 126 * 3,
            126 * 126,
            126,
            1,
        ]),
    );
    let b = ShapeStride::new(
        DimDyn::from(&[32, 16, 126, 126]),
        DimDyn::from(&[128 * 128 * 16, 128 * 128, 128, 1]),
    );

    c.bench_function("copy_from_logic", |b_| b_.iter(|| bench_all(a, b)));
}

criterion::criterion_group!(benches, bench, max_index_bench, copy_apply_index_bench);
criterion::criterion_main!(benches);

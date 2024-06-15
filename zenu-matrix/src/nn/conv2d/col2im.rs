use crate::{
    device::Device,
    dim::DimDyn,
    matrix::{Matrix, Owned, Ref},
    num::Num,
    slice_dynamic,
};

pub(crate) fn col2im<T: Num, D: Device>(
    col: Matrix<Ref<&T>, DimDyn, D>,
    img_shape: [usize; 4],
    kernel_size: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
) -> Matrix<Owned<T>, DimDyn, D> {
    let (batch_size, c, h, w) = (img_shape[0], img_shape[1], img_shape[2], img_shape[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = pad;
    let (oh, ow) = ((h + 2 * ph - kh) / sh + 1, (w + 2 * pw - kw) / sw + 1);

    let mut img =
        Matrix::<_, DimDyn, _>::zeros([batch_size, c, h + 2 * ph + sh - 1, w + 2 * pw + sw - 1]);

    for j in 0..kh {
        let j_lim = j + sh * oh;
        for i in 0..kw {
            let i_lim = i + sw * ow;
            let col_ref = col.to_ref();
            let col_ref = col_ref.slice_dyn(slice_dynamic!(.., .., j, i, .., ..));

            let mut img_slice = img.to_ref_mut().slice_mut_dyn(slice_dynamic!(
                ..,
                ..,
                j..j_lim;sh,
                i..i_lim;sw
            ));
            img_slice += col_ref;
        }
    }

    let img = img.slice_dyn(slice_dynamic!(.., .., ph..ph + h, pw..pw + w));
    img.new_matrix()
}

#[cfg(test)]
mod col2im {
    use crate::{
        device::cpu::Cpu,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    use super::col2im;

    #[test]
    fn col2im_small() {
        let col = (1..=1350).map(|x| x as f32).collect::<Vec<f32>>();
        let col = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(col, &[2, 3, 3, 3, 5, 5]);
        let img_shape = [2, 3, 5, 5];
        let kernel_shape = (3, 3);
        let stride = (1, 1);
        let pad = (1, 1);
        let img = col2im(col.to_ref(), img_shape, kernel_shape, stride, pad);
        let ans = vec![
            216, 402, 408, 414, 328, 564, 963, 972, 981, 732, 594, 1008, 1017, 1026, 762, 624,
            1053, 1062, 1071, 792, 576, 942, 948, 954, 688, 1116, 1752, 1758, 1764, 1228, 1914,
            2988, 2997, 3006, 2082, 1944, 3033, 3042, 3051, 2112, 1974, 3078, 3087, 3096, 2142,
            1476, 2292, 2298, 2304, 1588, 2016, 3102, 3108, 3114, 2128, 3264, 5013, 5022, 5031,
            3432, 3294, 5058, 5067, 5076, 3462, 3324, 5103, 5112, 5121, 3492, 2376, 3642, 3648,
            3654, 2488, 2916, 4452, 4458, 4464, 3028, 4614, 7038, 7047, 7056, 4782, 4644, 7083,
            7092, 7101, 4812, 4674, 7128, 7137, 7146, 4842, 3276, 4992, 4998, 5004, 3388, 3816,
            5802, 5808, 5814, 3928, 5964, 9063, 9072, 9081, 6132, 5994, 9108, 9117, 9126, 6162,
            6024, 9153, 9162, 9171, 6192, 4176, 6342, 6348, 6354, 4288, 4716, 7152, 7158, 7164,
            4828, 7314, 11088, 11097, 11106, 7482, 7344, 11133, 11142, 11151, 7512, 7374, 11178,
            11187, 11196, 7542, 5076, 7692, 7698, 7704, 5188,
        ]
        .iter()
        .map(|&x| x as f32)
        .collect::<Vec<f32>>();
        let ans = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(ans, &[2, 3, 5, 5]);
        assert!((img - ans).asum() < 1e-6);
    }
}

use crate::dim::{DimDyn, DimTrait};

pub(super) fn conv_output_shape(
    input: DimDyn,
    filter: DimDyn,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
) -> DimDyn {
    // 入力次元数のチェック
    // conv1d: input = [N, C_in, L], filter = [C_out, C_in, K]
    // conv2d: input = [N, C_in, H, W], filter = [C_out, C_in, K_h, K_w]
    // それ以外も想定可能だが、ここでは1D/2Dを想定とのこと
    assert!(
        input.len() == filter.len(),
        "Input and filter must have the same number of dimensions."
    );
    assert!(
        input.len() >= 3 && input.len() <= 4,
        "This function currently supports conv1d or conv2d only."
    );
    assert!(
        stride.len() == input.len() - 2,
        "Stride length must match the number of spatial dimensions."
    );
    assert!(
        padding.len() == input.len() - 2,
        "Padding length must match the number of spatial dimensions."
    );
    assert!(
        dilation.len() == input.len() - 2,
        "Dilation length must match the number of spatial dimensions."
    );

    let mut output = DimDyn::default();
    output.push_dim(input[0]);
    output.push_dim(filter[0]);

    for i in 2..input.len() {
        let in_size = input[i];
        let kernel_size = filter[i];
        let (strd, pad, dil) = (stride[i - 2], padding[i - 2], dilation[i - 2]);
        let out_size = conv_dim_out_size(in_size, kernel_size, pad, strd, dil);
        output.push_dim(out_size);
    }

    output
}

/// 1次元方向の出力サイズを計算するためのヘルパー関数
/// `out_size` = ((`in_size` + 2*`pad` - `dil`*(`kernel_size`-1) - 1) / `stride`) + 1
pub(super) fn conv_dim_out_size(
    in_size: usize,
    kernel_size: usize,
    pad: usize,
    stride: usize,
    dilation: usize,
) -> usize {
    ((in_size + 2 * pad - dilation * (kernel_size - 1) - 1) / stride) + 1
}

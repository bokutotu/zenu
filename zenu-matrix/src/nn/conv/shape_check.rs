use crate::dim::{DimDyn, DimTrait};

#[allow(clippy::module_name_repetitions)]
pub fn shape_check_2d(
    input_shape: DimDyn,
    filter_shape: DimDyn,
    output_shape: DimDyn,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
) {
    assert_eq!(input_shape.len(), 4, "Input shape must have 4 dimensions.");
    assert_eq!(
        filter_shape.len(),
        4,
        "Filter shape must have 4 dimensions."
    );
    assert_eq!(stride.len(), 2, "Stride must have 2 dimensions.");
    assert_eq!(padding.len(), 2, "Padding must have 2 dimensions.");
    assert_eq!(dilation.len(), 2, "Dilation must have 2 dimensions.");
    assert_eq!(
        dilation.len(),
        stride.len(),
        "Dilation length must match the number of spatial dimensions."
    );

    let h_out_expected =
        (input_shape[2] + 2 * padding[0] - dilation[0] * (filter_shape[2] - 1) - 1) / stride[0] + 1;
    let w_out_expected =
        (input_shape[3] + 2 * padding[1] - dilation[1] * (filter_shape[3] - 1) - 1) / stride[1] + 1;

    assert_eq!(
        h_out_expected, output_shape[2],
        "Output height mismatch: expected {}, got {}",
        h_out_expected, output_shape[2]
    );
    assert_eq!(
        w_out_expected, output_shape[3],
        "Output width mismatch: expected {}, got {}",
        w_out_expected, output_shape[3]
    );

    // channel数が一致しているか
    assert_eq!(
        input_shape[1], filter_shape[1],
        "Input and filter channel count must match."
    );
    // outputのchannel数が一致しているか
    assert_eq!(
        filter_shape[0], output_shape[1],
        "Filter and output channel count must match."
    );
}

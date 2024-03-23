use zenu_matrix::dim::{DimDyn, DimTrait};

pub fn conv2d_out_shape<D: DimTrait>(
    input: D,
    output_channel: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> DimDyn {
    // input is four dimension [batch, input_c, input_h, input_w]
    // kernel is four dimension [output_c, input_c, kernel_h, kernel_w]
    // stride is two dimension [stride_h, stride_w]
    // padding is two dimension [padding_h, padding_w]
    // output is four dimension [batch, output_c, output_h, output_w]
    let input = input.slice();
    let input_h = input[2];
    let input_w = input[3];
    let kernel_h = kernel.0;

    let output_h = (input_h + 2 * padding.0 - kernel_h) / stride.0 + 1;
    let output_w = (input_w + 2 * padding.1 - kernel.1) / stride.1 + 1;

    DimDyn::from([input[0], output_channel, output_h, output_w])
}

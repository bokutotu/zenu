use crate::{
    device::cpu::Cpu,
    dim::DimDyn,
    matrix::{Matrix, Ref},
    nn::conv::interface::ConvBias,
    num::Num,
};

impl ConvBias for Cpu {
    fn conv2d_bias<T: Num>(
        input: Matrix<Ref<&T>, DimDyn, Self>,
        bias: Matrix<Ref<&T>, DimDyn, Self>,
        mut output: Matrix<Ref<&mut T>, DimDyn, Self>,
    ) {
        output.add_array(&input, &bias);
    }

    fn conv2d_bias_bkwd<T: Num>(
        d_output: Matrix<Ref<&T>, DimDyn, Self>,
        d_bias: Matrix<Ref<&mut T>, DimDyn, Self>,
    ) {
        let dy_0 = d_output.sum(0, true);
        let dy_0_2 = dy_0.to_ref().sum(2, true);
        d_bias.copy_from(&dy_0_2.to_ref().sum(3, true));
    }
}

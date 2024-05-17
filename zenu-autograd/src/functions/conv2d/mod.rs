use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    dim::DimTrait,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    num::Num,
    operation::{
        copy_from::CopyFrom,
        mul::Gemm,
        reshape::{Reshape, ReshapeMut},
        transpose::{Transpose, TransposeInplace},
    },
};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

use self::{
    conv2d_impl::{conv2d_inner, conv2d_out_size},
    deconv2_impl::{deconv2d_inner, deconv2d_out_size},
    im2col::{im2col, Im2ColRes},
};

use super::sum::sum;

mod col2im;
mod conv2d_impl;
mod deconv2_impl;
mod im2col;

struct Conv2d<T: Num> {
    kernel: Variable<T>,
    bias: Option<Variable<T>>,
    input: Variable<T>,
    stride: (usize, usize),
    pad: (usize, usize),
    output: VariableWeak<T>,
}

struct Deconv2d<T: Num> {
    kernel: Variable<T>,
    bias: Option<Variable<T>>,
    input: Variable<T>,
    stride: (usize, usize),
    pad: (usize, usize),
    output: VariableWeak<T>,
}

struct Conv2dGrad<T: Num> {
    kernel: Variable<T>,
    input: Variable<T>,
    gradient_output: Variable<T>,
    stride: (usize, usize),
    pad: (usize, usize),
    output: VariableWeak<T>,
}

impl<T: Num, D: Device> Function<T> for Conv2d<T> {
    fn forward(&self) {
        let output = conv2d_inner(
            self.input.get_data().to_view(),
            self.kernel.get_data().to_view(),
            self.bias.clone().map(|x| x.get_data()),
            self.pad,
            self.stride,
        );
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .copy_from(&output);
    }

    fn backward(&self) {
        let g_input = deconv2d(
            self.output.upgrade().unwrap().get_grad().unwrap(),
            self.kernel.clone(),
            None,
            self.stride,
            self.pad,
        );
        let gw = conv2d_grad(
            self.input.clone(),
            self.output.upgrade().unwrap().get_grad().unwrap(),
            self.kernel.clone(),
            self.stride,
            self.pad,
        );
        self.input.set_grad(g_input);
        self.kernel.set_grad(gw);

        if let Some(bias) = &self.bias {
            let output_grad = self.output.upgrade().unwrap().get_grad().unwrap();
            let output_grad = sum(output_grad, 0, true);
            output_grad.set_name("conv2d_bias_grad_intermidiate_1");
            let output_grad = sum(output_grad, 2, true);
            output_grad.set_name("conv2d_bias_grad_intermidiate_2");
            let output_grad = sum(output_grad, 3, true);
            output_grad.set_name("conv2d_bias_grad");
            bias.set_grad(output_grad);
        }
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        match self.bias.clone() {
            Some(bias) => vec![self.kernel.clone(), self.input.clone(), bias],
            None => vec![self.kernel.clone(), self.input.clone()],
        }
    }
}

impl<T: Num, D: Device> Function<T> for Deconv2d<T> {
    fn forward(&self) {
        let output = deconv2d_inner(
            self.input.get_data().to_view(),
            self.kernel.get_data().to_view(),
            self.bias.clone().map(|x| x.get_data()),
            self.pad,
            self.stride,
        );
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .copy_from(&output);
    }

    fn backward(&self) {
        let g_input = conv2d(
            self.output.upgrade().unwrap().get_grad().unwrap(),
            self.kernel.clone(),
            self.bias.clone(),
            self.stride,
            self.pad,
        );
        let gw = conv2d_grad(
            self.output.upgrade().unwrap().get_grad().unwrap(),
            self.input.clone(),
            self.kernel.clone(),
            self.stride,
            self.pad,
        );
        self.input.set_grad(g_input);
        self.kernel.set_grad(gw);

        if let Some(bias) = self.bias.clone() {
            let output_grad = self.output.upgrade().unwrap().get_grad().unwrap();
            let output_grad = sum(output_grad, 0, true);
            let output_grad = sum(output_grad, 2, true);
            let output_grad = sum(output_grad, 3, true);
            bias.set_grad(output_grad);
        }
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        match self.bias.clone() {
            Some(bias) => vec![self.kernel.clone(), self.input.clone(), bias],
            None => vec![self.kernel.clone(), self.input.clone()],
        }
    }
}

impl<T: Num, D: Device> Function<T> for Conv2dGrad<T> {
    fn forward(&self) {
        let input = self.input.get_data();
        let kernel_shape = self.kernel.get_data().shape();
        let Im2ColRes { mut col, .. } = im2col(
            input.to_view(),
            (kernel_shape[2], kernel_shape[3]),
            self.stride,
            self.pad,
        );

        col.transpose();

        let gradient_output = self.gradient_output.get_data();
        let gradient_output = gradient_output.transpose_swap_index_inplace(0, 1);
        let gradient_output_shape = gradient_output.shape();
        let gradient_output = gradient_output.reshape_new_matrix([
            gradient_output_shape[0],
            gradient_output_shape.num_elm() / gradient_output_shape[0],
        ]);

        let output = self.output.upgrade().unwrap();
        output
            .get_data_mut()
            .to_view_mut()
            .reshape_mut([gradient_output.shape()[0], col.shape()[1]])
            .gemm(gradient_output, col);
    }

    fn backward(&self) {
        let g_input = deconv2d(
            self.gradient_output.clone(),
            self.output.upgrade().unwrap(),
            None,
            self.stride,
            self.pad,
        );
        let grad_grad_output = conv2d(
            self.input.clone(),
            self.output.upgrade().unwrap(),
            None,
            self.stride,
            self.pad,
        );
        self.input.set_grad(g_input);
        self.gradient_output.set_grad(grad_grad_output);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![
            self.kernel.clone(),
            self.input.clone(),
            self.gradient_output.clone(),
        ]
    }
}

pub fn conv2d<T: Num>(
    x: Variable<T>,
    kernel: Variable<T>,
    bias: Option<Variable<T>>,
    stride: (usize, usize),
    pad: (usize, usize),
) -> Variable<T> {
    let x_shape = x.get_data().shape();
    let kernel_shape = kernel.get_data().shape();
    let out_shape = conv2d_out_size(x_shape.slice(), kernel_shape.slice(), pad, stride);
    let output = zeros(out_shape);
    let conv2d = Conv2d {
        kernel,
        input: x,
        bias,
        stride,
        pad,
        output: output.clone().downgrade(),
    };

    conv2d.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(conv2d))));

    output
}

pub fn deconv2d<T: Num>(
    x: Variable<T>,
    kernel: Variable<T>,
    bias: Option<Variable<T>>,
    stride: (usize, usize),
    pad: (usize, usize),
) -> Variable<T> {
    let x_shape = x.get_data().shape();
    let kernel_shape = kernel.get_data().shape();
    let out_shape = deconv2d_out_size(x_shape.slice(), kernel_shape.slice(), pad, stride);
    let output = zeros(out_shape);
    let deconv2d = Deconv2d {
        kernel,
        input: x,
        stride,
        bias,
        pad,
        output: output.clone().downgrade(),
    };
    deconv2d.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(deconv2d))));
    output
}

pub fn conv2d_grad<T: Num>(
    x: Variable<T>,
    gradient_output: Variable<T>,
    kernel: Variable<T>,
    stride: (usize, usize),
    pad: (usize, usize),
) -> Variable<T> {
    let output = zeros(kernel.get_data().shape());
    let conv2d_grad = Conv2dGrad {
        kernel,
        input: x,
        gradient_output,
        stride,
        pad,
        output: output.clone().downgrade(),
    };
    conv2d_grad.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(conv2d_grad))));
    output
}

#[cfg(test)]
mod conv2d {
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use crate::creator::from_vec::from_vec;

    #[test]
    fn conv2d_2x3x5x5_image_4x3x3x3_kernel_1x1_stride_1x1_padding() {
        let kernel = (1..(4 * 3 * 3 * 3 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let kernel = from_vec(kernel, [4, 3, 3, 3]);
        let image = (1..(2 * 3 * 5 * 5 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let image = from_vec(image, [2, 3, 5, 5]);
        let output = super::conv2d(image.clone(), kernel.clone(), None, (1, 1), (1, 1));
        output.backward();

        let output_ans = vec![
            7416., 11010., 11289., 11568., 7608., 11106., 16434., 16812., 17190., 11268., 12411.,
            18324., 18702., 19080., 12483., 13716., 20214., 20592., 20970., 13698., 8712., 12792.,
            13017., 13242., 8616., 16812., 25347., 26112., 26877., 17976., 26415., 39762., 40869.,
            41976., 28035., 30150., 45297., 46404., 47511., 31680., 33885., 50832., 51939., 53046.,
            35325., 22968., 34419., 35130., 35841., 23844., 26208., 39684., 40935., 42186., 28344.,
            41724., 63090., 64926., 66762., 44802., 47889., 72270., 74106., 75942., 50877., 54054.,
            81450., 83286., 85122., 56952., 37224., 56046., 57243., 58440., 39072., 35604., 54021.,
            55758., 57495., 38712., 57033., 86418., 88983., 91548., 61569., 65628., 99243.,
            101808., 104373., 70074., 74223., 112068., 114633., 117198., 78579., 51480., 77673.,
            79356., 81039., 54300., 21816., 31935., 32214., 32493., 21108., 30681., 44784., 45162.,
            45540., 29493., 31986., 46674., 47052., 47430., 30708., 33291., 48564., 48942., 49320.,
            31923., 20412., 29667., 29892., 30117., 19416., 55512., 82722., 83487., 84252., 55776.,
            82440., 122787., 123894., 125001., 82710., 86175., 128322., 129429., 130536., 86355.,
            89910., 133857., 134964., 136071., 90000., 58968., 87744., 88455., 89166., 58944.,
            89208., 133509., 134760., 136011., 90444., 134199., 200790., 202626., 204462., 135927.,
            140364., 209970., 211806., 213642., 142002., 146529., 219150., 220986., 222822.,
            148077., 97524., 145821., 147018., 148215., 98472., 122904., 184296., 186033., 187770.,
            125112., 185958., 278793., 281358., 283923., 189144., 194553., 291618., 294183.,
            296748., 197649., 203148., 304443., 307008., 309573., 206154., 136080., 203898.,
            205581., 207264., 138000.,
        ];
        let output_ans = OwnedMatrixDyn::from_vec(output_ans, [2, 4, 5, 5]);

        let kernel_grad_ans = vec![
            1520., 1920., 1552., 2000., 2525., 2040., 1680., 2120., 1712., 2320., 2920., 2352.,
            3000., 3775., 3040., 2480., 3120., 2512., 3120., 3920., 3152., 4000., 5025., 4040.,
            3280., 4120., 3312., 1520., 1920., 1552., 2000., 2525., 2040., 1680., 2120., 1712.,
            2320., 2920., 2352., 3000., 3775., 3040., 2480., 3120., 2512., 3120., 3920., 3152.,
            4000., 5025., 4040., 3280., 4120., 3312., 1520., 1920., 1552., 2000., 2525., 2040.,
            1680., 2120., 1712., 2320., 2920., 2352., 3000., 3775., 3040., 2480., 3120., 2512.,
            3120., 3920., 3152., 4000., 5025., 4040., 3280., 4120., 3312., 1520., 1920., 1552.,
            2000., 2525., 2040., 1680., 2120., 1712., 2320., 2920., 2352., 3000., 3775., 3040.,
            2480., 3120., 2512., 3120., 3920., 3152., 4000., 5025., 4040., 3280., 4120., 3312.,
        ];
        let kernel_grad_ans = OwnedMatrixDyn::from_vec(kernel_grad_ans, [4, 3, 3, 3]);

        let image_grad_ans = vec![
            696., 1056., 1056., 1056., 712., 1080., 1638., 1638., 1638., 1104., 1080., 1638.,
            1638., 1638., 1104., 1080., 1638., 1638., 1638., 1104., 744., 1128., 1128., 1128.,
            760., 840., 1272., 1272., 1272., 856., 1296., 1962., 1962., 1962., 1320., 1296., 1962.,
            1962., 1962., 1320., 1296., 1962., 1962., 1962., 1320., 888., 1344., 1344., 1344.,
            904., 984., 1488., 1488., 1488., 1000., 1512., 2286., 2286., 2286., 1536., 1512.,
            2286., 2286., 2286., 1536., 1512., 2286., 2286., 2286., 1536., 1032., 1560., 1560.,
            1560., 1048., 696., 1056., 1056., 1056., 712., 1080., 1638., 1638., 1638., 1104.,
            1080., 1638., 1638., 1638., 1104., 1080., 1638., 1638., 1638., 1104., 744., 1128.,
            1128., 1128., 760., 840., 1272., 1272., 1272., 856., 1296., 1962., 1962., 1962., 1320.,
            1296., 1962., 1962., 1962., 1320., 1296., 1962., 1962., 1962., 1320., 888., 1344.,
            1344., 1344., 904., 984., 1488., 1488., 1488., 1000., 1512., 2286., 2286., 2286.,
            1536., 1512., 2286., 2286., 2286., 1536., 1512., 2286., 2286., 2286., 1536., 1032.,
            1560., 1560., 1560., 1048.,
        ];
        let image_grad_ans = OwnedMatrixDyn::from_vec(image_grad_ans, [2, 3, 5, 5]);

        let output_mat = output.get_data();
        let kernel_grad_mat = kernel.get_grad().unwrap().get_data();
        let image_grad_mat = image.get_grad().unwrap().get_data();
        assert!((output_mat - output_ans).asum() < 1e-6);
        assert!((image_grad_mat - image_grad_ans).asum() < 1e-6);
        assert!((kernel_grad_mat - kernel_grad_ans).asum() < 1e-6);
    }

    #[test]
    fn deconv2d_2x3x5x5_image_4x3x3x3_kernel_1x1_stride_1x1_padding() {
        let kernel = (1..(3 * 4 * 3 * 3 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let kernel = from_vec(kernel, [3, 4, 3, 3]);
        let image = (1..(2 * 3 * 5 * 5 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let image = from_vec(image, [2, 3, 5, 5]);
        let output = super::deconv2d(image.clone(), kernel.clone(), None, (2, 2), (1, 1));
        output.backward();
        let output_ans = vec![
            4998, 10116, 5121, 10362, 5244, 10608, 5367, 10854, 5490, 10566, 21372, 10812, 21864,
            11058, 22356, 11304, 22848, 11550, 5613, 11346, 5736, 11592, 5859, 11838, 5982, 12084,
            6105, 11796, 23832, 12042, 24324, 12288, 24816, 12534, 25308, 12780, 6228, 12576, 6351,
            12822, 6474, 13068, 6597, 13314, 6720, 13026, 26292, 13272, 26784, 13518, 27276, 13764,
            27768, 14010, 6843, 13806, 6966, 14052, 7089, 14298, 7212, 14544, 7335, 14256, 28752,
            14502, 29244, 14748, 29736, 14994, 30228, 15240, 7458, 15036, 7581, 15282, 7704, 15528,
            7827, 15774, 7950, 5700, 11547, 5850, 11847, 6000, 12147, 6150, 12447, 6300, 12105,
            24504, 12405, 25104, 12705, 25704, 13005, 26304, 13305, 6450, 13047, 6600, 13347, 6750,
            13647, 6900, 13947, 7050, 13605, 27504, 13905, 28104, 14205, 28704, 14505, 29304,
            14805, 7200, 14547, 7350, 14847, 7500, 15147, 7650, 15447, 7800, 15105, 30504, 15405,
            31104, 15705, 31704, 16005, 32304, 16305, 7950, 16047, 8100, 16347, 8250, 16647, 8400,
            16947, 8550, 16605, 33504, 16905, 34104, 17205, 34704, 17505, 35304, 17805, 8700,
            17547, 8850, 17847, 9000, 18147, 9150, 18447, 9300, 6402, 12978, 6579, 13332, 6756,
            13686, 6933, 14040, 7110, 13644, 27636, 13998, 28344, 14352, 29052, 14706, 29760,
            15060, 7287, 14748, 7464, 15102, 7641, 15456, 7818, 15810, 7995, 15414, 31176, 15768,
            31884, 16122, 32592, 16476, 33300, 16830, 8172, 16518, 8349, 16872, 8526, 17226, 8703,
            17580, 8880, 17184, 34716, 17538, 35424, 17892, 36132, 18246, 36840, 18600, 9057,
            18288, 9234, 18642, 9411, 18996, 9588, 19350, 9765, 18954, 38256, 19308, 38964, 19662,
            39672, 20016, 40380, 20370, 9942, 20058, 10119, 20412, 10296, 20766, 10473, 21120,
            10650, 7104, 14409, 7308, 14817, 7512, 15225, 7716, 15633, 7920, 15183, 30768, 15591,
            31584, 15999, 32400, 16407, 33216, 16815, 8124, 16449, 8328, 16857, 8532, 17265, 8736,
            17673, 8940, 17223, 34848, 17631, 35664, 18039, 36480, 18447, 37296, 18855, 9144,
            18489, 9348, 18897, 9552, 19305, 9756, 19713, 9960, 19263, 38928, 19671, 39744, 20079,
            40560, 20487, 41376, 20895, 10164, 20529, 10368, 20937, 10572, 21345, 10776, 21753,
            10980, 21303, 43008, 21711, 43824, 22119, 44640, 22527, 45456, 22935, 11184, 22569,
            11388, 22977, 11592, 23385, 11796, 23793, 12000, 14223, 28566, 14346, 28812, 14469,
            29058, 14592, 29304, 14715, 29016, 58272, 29262, 58764, 29508, 59256, 29754, 59748,
            30000, 14838, 29796, 14961, 30042, 15084, 30288, 15207, 30534, 15330, 30246, 60732,
            30492, 61224, 30738, 61716, 30984, 62208, 31230, 15453, 31026, 15576, 31272, 15699,
            31518, 15822, 31764, 15945, 31476, 63192, 31722, 63684, 31968, 64176, 32214, 64668,
            32460, 16068, 32256, 16191, 32502, 16314, 32748, 16437, 32994, 16560, 32706, 65652,
            32952, 66144, 33198, 66636, 33444, 67128, 33690, 16683, 33486, 16806, 33732, 16929,
            33978, 17052, 34224, 17175, 16950, 34047, 17100, 34347, 17250, 34647, 17400, 34947,
            17550, 34605, 69504, 34905, 70104, 35205, 70704, 35505, 71304, 35805, 17700, 35547,
            17850, 35847, 18000, 36147, 18150, 36447, 18300, 36105, 72504, 36405, 73104, 36705,
            73704, 37005, 74304, 37305, 18450, 37047, 18600, 37347, 18750, 37647, 18900, 37947,
            19050, 37605, 75504, 37905, 76104, 38205, 76704, 38505, 77304, 38805, 19200, 38547,
            19350, 38847, 19500, 39147, 19650, 39447, 19800, 39105, 78504, 39405, 79104, 39705,
            79704, 40005, 80304, 40305, 19950, 40047, 20100, 40347, 20250, 40647, 20400, 40947,
            20550, 19677, 39528, 19854, 39882, 20031, 40236, 20208, 40590, 20385, 40194, 80736,
            40548, 81444, 40902, 82152, 41256, 82860, 41610, 20562, 41298, 20739, 41652, 20916,
            42006, 21093, 42360, 21270, 41964, 84276, 42318, 84984, 42672, 85692, 43026, 86400,
            43380, 21447, 43068, 21624, 43422, 21801, 43776, 21978, 44130, 22155, 43734, 87816,
            44088, 88524, 44442, 89232, 44796, 89940, 45150, 22332, 44838, 22509, 45192, 22686,
            45546, 22863, 45900, 23040, 45504, 91356, 45858, 92064, 46212, 92772, 46566, 93480,
            46920, 23217, 46608, 23394, 46962, 23571, 47316, 23748, 47670, 23925, 22404, 45009,
            22608, 45417, 22812, 45825, 23016, 46233, 23220, 45783, 91968, 46191, 92784, 46599,
            93600, 47007, 94416, 47415, 23424, 47049, 23628, 47457, 23832, 47865, 24036, 48273,
            24240, 47823, 96048, 48231, 96864, 48639, 97680, 49047, 98496, 49455, 24444, 49089,
            24648, 49497, 24852, 49905, 25056, 50313, 25260, 49863, 100128, 50271, 100944, 50679,
            101760, 51087, 102576, 51495, 25464, 51129, 25668, 51537, 25872, 51945, 26076, 52353,
            26280, 51903, 104208, 52311, 105024, 52719, 105840, 53127, 106656, 53535, 26484, 53169,
            26688, 53577, 26892, 53985, 27096, 54393, 27300,
        ]
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();
        let output_ans = OwnedMatrixDyn::from_vec(output_ans, [2, 4, 9, 9]);
        assert!((output.get_data() - output_ans).asum() < 1e-6);

        let input_grad_ans = vec![
            328.0, 480.0, 480.0, 480.0, 312.0, 456.0, 666.0, 666.0, 666.0, 432.0, 456.0, 666.0,
            666.0, 666.0, 432.0, 456.0, 666.0, 666.0, 666.0, 432.0, 280.0, 408.0, 408.0, 408.0,
            264.0, 904.0, 1344.0, 1344.0, 1344.0, 888.0, 1320.0, 1962.0, 1962.0, 1962.0, 1296.0,
            1320.0, 1962.0, 1962.0, 1962.0, 1296.0, 1320.0, 1962.0, 1962.0, 1962.0, 1296.0, 856.0,
            1272.0, 1272.0, 1272.0, 840.0, 1480.0, 2208.0, 2208.0, 2208.0, 1464.0, 2184.0, 3258.0,
            3258.0, 3258.0, 2160.0, 2184.0, 3258.0, 3258.0, 3258.0, 2160.0, 2184.0, 3258.0, 3258.0,
            3258.0, 2160.0, 1432.0, 2136.0, 2136.0, 2136.0, 1416.0, 328.0, 480.0, 480.0, 480.0,
            312.0, 456.0, 666.0, 666.0, 666.0, 432.0, 456.0, 666.0, 666.0, 666.0, 432.0, 456.0,
            666.0, 666.0, 666.0, 432.0, 280.0, 408.0, 408.0, 408.0, 264.0, 904.0, 1344.0, 1344.0,
            1344.0, 888.0, 1320.0, 1962.0, 1962.0, 1962.0, 1296.0, 1320.0, 1962.0, 1962.0, 1962.0,
            1296.0, 1320.0, 1962.0, 1962.0, 1962.0, 1296.0, 856.0, 1272.0, 1272.0, 1272.0, 840.0,
            1480.0, 2208.0, 2208.0, 2208.0, 1464.0, 2184.0, 3258.0, 3258.0, 3258.0, 2160.0, 2184.0,
            3258.0, 3258.0, 3258.0, 2160.0, 2184.0, 3258.0, 3258.0, 3258.0, 2160.0, 1432.0, 2136.0,
            2136.0, 2136.0, 1416.0,
        ];
        let input_grad_ans = OwnedMatrixDyn::from_vec(input_grad_ans, [2, 3, 5, 5]);
        assert!((image.get_grad().unwrap().get_data() - input_grad_ans).asum() < 1e-6);

        let kernel_grad = vec![
            1712.0, 2120.0, 1680.0, 2040.0, 2525.0, 2000.0, 1552.0, 1920.0, 1520.0, 1712.0, 2120.0,
            1680.0, 2040.0, 2525.0, 2000.0, 1552.0, 1920.0, 1520.0, 1712.0, 2120.0, 1680.0, 2040.0,
            2525.0, 2000.0, 1552.0, 1920.0, 1520.0, 1712.0, 2120.0, 1680.0, 2040.0, 2525.0, 2000.0,
            1552.0, 1920.0, 1520.0, 2512.0, 3120.0, 2480.0, 3040.0, 3775.0, 3000.0, 2352.0, 2920.0,
            2320.0, 2512.0, 3120.0, 2480.0, 3040.0, 3775.0, 3000.0, 2352.0, 2920.0, 2320.0, 2512.0,
            3120.0, 2480.0, 3040.0, 3775.0, 3000.0, 2352.0, 2920.0, 2320.0, 2512.0, 3120.0, 2480.0,
            3040.0, 3775.0, 3000.0, 2352.0, 2920.0, 2320.0, 3312.0, 4120.0, 3280.0, 4040.0, 5025.0,
            4000.0, 3152.0, 3920.0, 3120.0, 3312.0, 4120.0, 3280.0, 4040.0, 5025.0, 4000.0, 3152.0,
            3920.0, 3120.0, 3312.0, 4120.0, 3280.0, 4040.0, 5025.0, 4000.0, 3152.0, 3920.0, 3120.0,
            3312.0, 4120.0, 3280.0, 4040.0, 5025.0, 4000.0, 3152.0, 3920.0, 3120.0,
        ];
        let kernel_grad = OwnedMatrixDyn::from_vec(kernel_grad, [3, 4, 3, 3]);
        assert!((kernel.get_grad().unwrap().get_data() - kernel_grad).asum() < 1e-6);
    }

    #[test]
    fn conv2d_2x3x5x5_image_4x3x3x3_kernel_1x1_stride_1x1_padding_with_bias() {
        let kernel = (1..(4 * 3 * 3 * 3 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let kernel = from_vec(kernel, [4, 3, 3, 3]);
        let image = (1..(2 * 3 * 5 * 5 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let image = from_vec(image, [2, 3, 5, 5]);
        let bias = (1..5).map(|x| x as f32).collect::<Vec<f32>>();
        let bias = from_vec(bias, [1, 4, 1, 1]);
        let output = super::conv2d(image.clone(), kernel.clone(), Some(bias), (1, 1), (1, 1));
        output.backward();
    }
}

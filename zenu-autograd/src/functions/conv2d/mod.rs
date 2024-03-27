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
    im2col::im2col,
};

mod col2im;
mod conv2d_impl;
mod deconv2_impl;
mod im2col;

struct Conv2d<T: Num> {
    kernel: Variable<T>,
    input: Variable<T>,
    stride: (usize, usize),
    pad: (usize, usize),
    output: VariableWeak<T>,
}

struct Deconv2d<T: Num> {
    kernel: Variable<T>,
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

impl<T: Num> Function<T> for Conv2d<T> {
    fn forward(&self) {
        let output = conv2d_inner(
            self.input.get_data().to_view(),
            self.kernel.get_data().to_view(),
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
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.kernel.clone(), self.input.clone()]
    }
}

impl<T: Num> Function<T> for Deconv2d<T> {
    fn forward(&self) {
        let output = deconv2d_inner(
            self.input.get_data().to_view(),
            self.kernel.get_data().to_view(),
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
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.kernel.clone(), self.input.clone()]
    }
}

impl<T: Num> Function<T> for Conv2dGrad<T> {
    fn forward(&self) {
        let input = self.input.get_data();
        let kernel_shape = self.kernel.get_data().shape();
        let col = im2col(
            input.to_view(),
            (kernel_shape[2], kernel_shape[3]),
            self.stride,
            self.pad,
        );
        let gradient_output = self.gradient_output.get_data();
        let grad_output_shape = gradient_output.shape();
        let grad_output_num_elm = grad_output_shape.num_elm();
        let gradient_output_transpose = gradient_output.transpose_swap_index_inplace(0, 1);
        let mut gradient_output_transose_reshape = gradient_output_transpose.reshape([
            grad_output_shape[1],
            grad_output_num_elm / grad_output_shape[1],
        ]);
        gradient_output_transose_reshape.transpose();

        let output = self.output.upgrade().unwrap();
        let output_shape = output.get_data().shape();
        let mut output = output.get_data_mut();
        let mut output = output.reshape_mut([col.col.shape()[0], grad_output_shape[1]]);
        output
            .to_view_mut()
            .gemm(col.col.to_view(), gradient_output_transose_reshape);
        output.reshape(output_shape.slice());
    }

    fn backward(&self) {
        let g_input = deconv2d(
            self.gradient_output.clone(),
            self.output.upgrade().unwrap(),
            self.stride,
            self.pad,
        );
        let grad_grad_output = conv2d(
            self.input.clone(),
            self.output.upgrade().unwrap(),
            self.stride,
            self.pad,
        );
        self.input.set_grad(g_input);
        self.gradient_output.set_grad(grad_grad_output);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.kernel.clone(), self.input.clone()]
    }
}

pub fn conv2d<T: Num>(
    x: Variable<T>,
    kernel: Variable<T>,
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
    use zenu_matrix::{
        matrix::{MatrixBase, OwnedMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::asum::Asum,
    };

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
        let output = super::conv2d(image.clone(), kernel.clone(), (1, 1), (1, 1));
        println!("{:?}", output.get_data().shape());
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
}

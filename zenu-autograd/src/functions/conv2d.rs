use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::DimTrait,
    nn::conv2d::{
        conv2d_bckwd_data, conv2d_bckwd_filter, conv2d_forward, conv2d_out_size, deconv2d_out_size,
    },
    num::Num,
};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

struct Conv2d<T: Num, D: Device> {
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    y: VariableWeak<T, D>,
}

struct Deconv2d<T: Num, D: Device> {
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    y: VariableWeak<T, D>,
}

struct Conv2dBackward<T: Num, D: Device> {
    y_grad: Variable<T, D>,
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    filter_grad: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for Conv2d<T, D> {
    fn forward(&self) {
        let y = conv2d_forward(
            self.x.get_data().to_ref(),
            self.filter.get_data().to_ref(),
            // FIXME bias
            None,
            self.padding.0,
            self.padding.1,
            self.stride.0,
            self.stride.1,
            // FIXME use dilated
            1,
            1,
            None,
        );
        self.y
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&y);
    }

    fn backward(&self) {
        let gx = deconv2d(
            self.y.upgrade().unwrap().get_grad().unwrap(),
            self.filter.clone(),
            self.stride,
            self.padding,
            None,
        );

        let gfilter = conv2d_filter_grad(
            self.x.clone(),
            self.y.upgrade().unwrap().get_grad().unwrap(),
            self.stride,
            self.padding,
            self.filter.clone(),
        );

        self.x.set_grad(gx);
        self.filter.set_grad(gfilter);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.filter.clone()]
    }
}

impl<T: Num, D: Device> Function<T, D> for Deconv2d<T, D> {
    fn forward(&self) {
        let y = conv2d_bckwd_data(
            self.x.get_data().to_ref(),
            self.filter.get_data().to_ref(),
            self.padding.0,
            self.padding.1,
            self.stride.0,
            self.stride.1,
            1,
            1,
            None,
        );

        self.y
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&y);
    }

    fn backward(&self) {
        let gx = conv2d(
            self.y.upgrade().unwrap().clone(),
            self.filter.clone(),
            self.stride,
            self.padding,
            None,
        );

        let gfilter = conv2d_filter_grad(
            self.x.clone(),
            self.filter.clone(),
            self.stride,
            self.padding,
            self.filter.clone(),
        );

        self.x.set_grad(gx);
        self.filter.set_grad(gfilter);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.filter.clone()]
    }
}

impl<T: Num, D: Device> Function<T, D> for Conv2dBackward<T, D> {
    fn forward(&self) {
        let filter_grad = conv2d_bckwd_filter(
            self.x.get_data().to_ref(),
            self.y_grad.get_data().to_ref(),
            self.padding.0,
            self.padding.1,
            self.stride.0,
            self.stride.1,
            1,
            1,
            self.filter.get_data().shape(),
            None,
        );
        self.filter_grad
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&filter_grad);
    }

    fn backward(&self) {
        let gx = deconv2d(
            self.y_grad.clone(),
            self.filter.clone(),
            self.stride,
            self.padding,
            None,
        );

        let gfilter = conv2d(
            self.x.clone(),
            self.filter_grad.upgrade().unwrap(),
            self.stride,
            self.padding,
            None,
        );

        self.x.set_grad(gx);
        self.filter.set_grad(gfilter);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.filter.clone()]
    }
}

pub fn conv2d<T: Num, D: Device>(
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    bias: Option<Variable<T, D>>,
) -> Variable<T, D> {
    let conv2d_y_size = conv2d_out_size(
        x.get_data().shape().slice(),
        filter.get_data().shape().slice(),
        padding,
        stride,
    );
    let y = zeros(conv2d_y_size);
    let conv2d = Conv2d {
        x,
        filter,
        stride,
        padding,
        y: y.clone().downgrade(),
    };
    conv2d.forward();
    y.set_creator(Rc::new(RefCell::new(Box::new(conv2d))));
    match bias {
        Some(bias) => y + bias,
        None => y,
    }
}

pub fn deconv2d<T: Num, D: Device>(
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    bias: Option<Variable<T, D>>,
) -> Variable<T, D> {
    let deconv2d_y_size = deconv2d_out_size(
        x.get_data().shape().slice(),
        filter.get_data().shape().slice(),
        padding,
        stride,
    );
    let y = zeros(deconv2d_y_size);
    let deconv2d = Deconv2d {
        x,
        filter,
        stride,
        padding,
        y: y.clone().downgrade(),
    };
    deconv2d.forward();
    y.set_creator(Rc::new(RefCell::new(Box::new(deconv2d))));
    match bias {
        Some(bias) => y + bias,
        None => y,
    }
}

fn conv2d_filter_grad<T: Num, D: Device>(
    x: Variable<T, D>,
    y_grad: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    filter: Variable<T, D>,
) -> Variable<T, D> {
    let filter_grad = zeros(filter.get_data().shape().slice());
    let conv2d_bkwd_filter = Conv2dBackward {
        y_grad,
        x,
        filter,
        stride,
        padding,
        filter_grad: filter_grad.clone().downgrade(),
    };
    conv2d_bkwd_filter.forward();
    filter_grad.set_creator(Rc::new(RefCell::new(Box::new(conv2d_bkwd_filter))));
    filter_grad
}

#[cfg(test)]
mod conv2d {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::creator::from_vec::from_vec;

    use super::conv2d;

    fn conv2d_2x3x5x5_image_4x3x3x3_kernel_1x1_stride_1x1_padding<D: Device>() {
        let kernel = (1..(4 * 3 * 3 * 3 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let kernel = from_vec(kernel, [4, 3, 3, 3]);
        let image = (1..(2 * 3 * 5 * 5 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let image = from_vec(image, [2, 3, 5, 5]);
        let output = conv2d(image.clone(), kernel.clone(), (1, 1), (1, 1), None);

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
        let output_ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(output_ans, [2, 4, 5, 5]);
        assert_val_eq!(output.clone(), output_ans, 1e-6);

        output.backward();

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
        let kernel_grad_ans =
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(kernel_grad_ans, [4, 3, 3, 3]);

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
        let image_grad_ans =
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(image_grad_ans, [2, 3, 5, 5]);

        assert_val_eq_grad!(kernel, kernel_grad_ans, 1e-6);
        assert_val_eq_grad!(image, image_grad_ans, 1e-6);
    }
    run_test!(
        conv2d_2x3x5x5_image_4x3x3x3_kernel_1x1_stride_1x1_padding,
        cc,
        nn
    );
}

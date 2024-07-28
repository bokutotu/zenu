use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::{DimDyn, DimTrait},
    nn::conv2d::{
        conv2d_bckwd_data, conv2d_bckwd_data_bias, conv2d_bckwd_filter, conv2d_bias_add,
        conv2d_forward, conv2d_out_size, deconv2d_out_size, Conv2dConfig,
    },
    num::Num,
};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

pub struct Conv2dConfigsInner<T: Num> {
    pub conv2d_forward: Conv2dConfig<T>,
    pub deconv2d: Conv2dBckwdDataConfig<T>,
    pub conv2d_bkwdfilter: Conv2dBckwdFilterConfig<T>,
}

impl<T: Num> Conv2dConfigsInner<T> {
    pub fn new(
        input: DimDyn,
        output: DimDyn,
        filter: DimDyn,
        stride: (usize, usize),
        padding: (usize, usize),
        num_algo: usize,
    ) -> Self {
        let conv2d_forward = Conv2dConfig::new(
            input, output, filter, padding.0, padding.1, stride.0, stride.1, 1, 1, num_algo,
        );
        let deconv2d = Conv2dBckwdDataConfig::new(
            input, output, filter, padding.0, padding.1, stride.0, stride.1, 1, 1, num_algo,
        );
        let conv2d_bkwdfilter = Conv2dBckwdFilterConfig::new(
            input, output, filter, padding.0, padding.1, stride.0, stride.1, 1, 1, num_algo,
        );
        Self {
            conv2d_forward,
            deconv2d,
            conv2d_bkwdfilter,
        }
    }
}

#[derive(Clone)]
pub struct Conv2dConfigs<T: Num> {
    pub inner: Rc<Conv2dConfigsInner<T>>,
}

impl<T: Num> Conv2dConfigs<T> {
    pub fn new(
        input: DimDyn,
        output: DimDyn,
        filter: DimDyn,
        stride: (usize, usize),
        padding: (usize, usize),
        num_algo: usize,
    ) -> Self {
        Self {
            inner: Rc::new(Conv2dConfigsInner::new(
                input, output, filter, stride, padding, num_algo,
            )),
        }
    }

    pub fn get_conv2d_forward(&self) -> &Conv2dConfig<T> {
        &self.inner.conv2d_forward
    }

    pub fn get_deconv2d(&self) -> &Conv2dBckwdDataConfig<T> {
        &self.inner.deconv2d
    }

    pub fn get_conv2d_bkwdfilter(&self) -> &Conv2dBckwdFilterConfig<T> {
        &self.inner.conv2d_bkwdfilter
    }
}

struct Conv2d<T: Num, D: Device> {
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    y: VariableWeak<T, D>,
    config: Conv2dConfigs<T>,
}

struct Deconv2d<T: Num, D: Device> {
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    y: VariableWeak<T, D>,
    config: Conv2dConfigs<T>,
}

struct Conv2dBackward<T: Num, D: Device> {
    y_grad: Variable<T, D>,
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    filter_grad: VariableWeak<T, D>,
    config: Conv2dConfigs<T>,
}

struct Conv2dBiasAdd<T: Num, D: Device> {
    y: Variable<T, D>,
    bias: Variable<T, D>,
    output: VariableWeak<T, D>,
}

struct Conv2dBiasBackward<T: Num, D: Device> {
    y_grad: Variable<T, D>,
    bias_grad: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for Conv2d<T, D> {
    fn forward(&self) {
        let config = self.config.get_conv2d_forward();
        let y = conv2d_forward(
            self.x.get_data().to_ref(),
            self.filter.get_data().to_ref(),
            self.padding.0,
            self.padding.1,
            self.stride.0,
            self.stride.1,
            // FIXME use dilated
            1,
            1,
            // FIXME
            Some(config),
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
            Some(self.config.clone()),
        );

        let gfilter = conv2d_filter_grad(
            self.x.clone(),
            self.y.upgrade().unwrap().get_grad().unwrap(),
            self.stride,
            self.padding,
            self.filter.clone(),
            Some(self.config.clone()),
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
            Some(self.config.clone().get_deconv2d()),
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
            Some(self.config.clone()),
        );

        let gfilter = conv2d_filter_grad(
            self.x.clone(),
            self.filter.clone(),
            self.stride,
            self.padding,
            self.filter.clone(),
            Some(self.config.clone()),
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
            Some(self.config.clone().get_conv2d_bkwdfilter()),
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
            Some(self.config.clone()),
        );

        let gfilter = conv2d(
            self.x.clone(),
            self.filter_grad.upgrade().unwrap(),
            self.stride,
            self.padding,
            None,
            Some(self.config.clone()),
        );

        self.x.set_grad(gx);
        self.filter.set_grad(gfilter);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.filter.clone()]
    }
}

impl<T: Num, D: Device> Function<T, D> for Conv2dBiasAdd<T, D> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        conv2d_bias_add(
            self.y.get_data().to_ref(),
            self.bias.get_data().to_ref(),
            output.to_ref_mut(),
        )
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        self.y.set_grad(output_grad.clone());
        let bias_grad = conv2d_bias_backward(output_grad, self.bias.get_shape());
        self.bias.set_grad(bias_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.y.clone(), self.bias.clone()]
    }
}

impl<T: Num, D: Device> Function<T, D> for Conv2dBiasBackward<T, D> {
    fn forward(&self) {
        let bias_grad = self.bias_grad.upgrade().unwrap();
        let mut bias_grad = bias_grad.get_data_mut();
        conv2d_bckwd_data_bias(self.y_grad.get_data().to_ref(), bias_grad.to_ref_mut())
    }

    fn backward(&self) {
        unimplemented!()
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.y_grad.clone()]
    }
}

pub fn conv2d<T: Num, D: Device>(
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    bias: Option<Variable<T, D>>,
    config: Option<Conv2dConfigs<T>>,
) -> Variable<T, D> {
    let output_shape = conv2d_out_size(
        x.get_data().shape().slice(),
        filter.get_data().shape().slice(),
        padding,
        stride,
    );
    let output_shape = DimDyn::from(&output_shape);
    let config = config.unwrap_or_else(|| {
        Conv2dConfigs::new(
            x.get_data().shape(),
            output_shape,
            filter.get_data().shape(),
            stride,
            padding,
            1,
        )
    });
    let conv2d_y_size = conv2d_out_size(
        x.get_data().shape().slice(),
        filter.get_data().shape().slice(),
        padding,
        stride,
    );
    let y = alloc(conv2d_y_size);
    let conv2d = Conv2d {
        x,
        filter,
        stride,
        padding,
        y: y.clone().downgrade(),
        config,
    };
    conv2d.forward();
    y.set_creator(Rc::new(RefCell::new(Box::new(conv2d))));
    match bias {
        Some(bias) => conv2d_bias(y, bias),
        None => y,
    }
}

pub fn deconv2d<T: Num, D: Device>(
    x: Variable<T, D>,
    filter: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    bias: Option<Variable<T, D>>,
    config: Option<Conv2dConfigs<T>>,
) -> Variable<T, D> {
    let config = config.unwrap_or_else(|| {
        Conv2dConfigs::new(
            x.get_data().shape(),
            filter.get_data().shape(),
            filter.get_data().shape(),
            stride,
            padding,
            0,
        )
    });
    let deconv2d_y_size = deconv2d_out_size(
        x.get_data().shape().slice(),
        filter.get_data().shape().slice(),
        padding,
        stride,
    );
    let y = alloc(deconv2d_y_size);
    let deconv2d = Deconv2d {
        x,
        filter,
        stride,
        padding,
        y: y.clone().downgrade(),
        config,
    };
    deconv2d.forward();
    y.set_creator(Rc::new(RefCell::new(Box::new(deconv2d))));
    match bias {
        Some(bias) => conv2d_bias(y, bias),
        None => y,
    }
}

fn conv2d_filter_grad<T: Num, D: Device>(
    x: Variable<T, D>,
    y_grad: Variable<T, D>,
    stride: (usize, usize),
    padding: (usize, usize),
    filter: Variable<T, D>,
    config: Option<Conv2dConfigs<T>>,
) -> Variable<T, D> {
    let config = config.unwrap_or_else(|| {
        Conv2dConfigs::new(
            x.get_data().shape(),
            filter.get_data().shape(),
            filter.get_data().shape(),
            stride,
            padding,
            0,
        )
    });
    let filter_grad = alloc(filter.get_data().shape().slice());
    let conv2d_bkwd_filter = Conv2dBackward {
        y_grad,
        x,
        filter,
        stride,
        padding,
        filter_grad: filter_grad.clone().downgrade(),
        config,
    };
    conv2d_bkwd_filter.forward();
    filter_grad.set_creator(Rc::new(RefCell::new(Box::new(conv2d_bkwd_filter))));
    filter_grad
}

fn conv2d_bias<T: Num, D: Device>(y: Variable<T, D>, bias: Variable<T, D>) -> Variable<T, D> {
    let output = alloc(y.get_data().shape().slice());
    let conv2d_bias_add = Conv2dBiasAdd {
        y,
        bias,
        output: output.clone().downgrade(),
    };
    conv2d_bias_add.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(conv2d_bias_add))));
    output
}

fn conv2d_bias_backward<T: Num, D: Device>(
    y_grad: Variable<T, D>,
    shape: DimDyn,
) -> Variable<T, D> {
    let bias_grad = alloc(shape.slice());
    let conv2d_bias_backward = Conv2dBiasBackward {
        y_grad,
        bias_grad: bias_grad.clone().downgrade(),
    };
    conv2d_bias_backward.forward();
    bias_grad.set_creator(Rc::new(RefCell::new(Box::new(conv2d_bias_backward))));
    bias_grad
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
        let output = conv2d(image.clone(), kernel.clone(), (1, 1), (1, 1), None, None);

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

    fn conv2d_with_bias<D: Device>() {
        // Input shape [1, 2, 4, 4]
        // Weight Shape:  torch.Size([3, 2, 3, 3])
        // Input:  [-0.45032975, -0.5730771, -0.5553584, 0.59432304, 1.5419426, 0.5073344, -0.59103316, -0.5692481, 0.18855357, -0.069072686, -0.49492535, -1.4959149, -0.19383712, 0.44551218, 1.3252748, -1.629326, 2.0819554, 1.7067116, 2.3803675, -1.1256016, -0.3169981, -1.0924683, -0.0851943, -0.093348235, -0.7607072, -1.599082, 0.018486667, -0.7504268, 0.18540798, 0.62113833, 0.63818157, -0.24600095]
        // Output:  [-0.23461546, -0.7300762, -0.43270034, -0.27294904, -0.22774543, -0.165058, -0.018641159, 0.3037408, 0.639988, 0.96370995, 0.83733726, 0.09503274, 0.2800117, 0.18681201, -0.1966653, -0.1629044, 0.38531667, -0.44342816, 0.3783964, -0.6913619, 0.032867774, -0.6861952, -0.4665652, -0.46068507, -0.15072693, -0.5189232, -0.3886519, -0.25197878, 0.075091064, 0.38447255, 0.032681137, 0.43084383, -0.1430219, 0.050474584, -0.48825723, -0.178395, 0.08189282, 0.32435876, 0.248155, -0.03715375, -0.008421779, -0.29653978, 0.3233031, 0.47368968, -0.22343802, -0.29611385, 0.045936823, -0.23833159]
        // Gradient:  [-0.06312838, 0.05240719, 0.05240719, 0.21505278, -0.07415994, 0.063570745, 0.063570745, 0.22900042, -0.07415994, 0.063570745, 0.063570745, 0.22900042, -0.0014246926, 0.13951382, 0.13951382, 0.005797662, -0.73124456, -0.7982433, -0.7982433, -0.098860174, -0.57463914, -0.689119, -0.689119, -0.12428501, -0.57463914, -0.689119, -0.689119, -0.12428501, -0.22594097, -0.37261552, -0.37261552, -0.085577406]
        // Weight:  [-0.0017646605, 0.12644097, -0.1939936, -0.1734625, -0.090781756, 0.063205294, -0.0046700113, 0.18688585, -0.020917172, 0.06236978, -0.071232304, -0.046330906, -0.2251778, -0.15610139, -0.09716192, 0.008731253, 0.0931814, 0.14142673, -0.15979224, -0.10263957, 0.0856111, 0.19572432, -0.048507567, 0.17637877, -0.03799128, 0.024940623, 0.21342279, -0.218654, -0.14838351, -0.05967162, -0.09187673, 0.20364694, -0.1527774, -0.1085015, -0.16467114, -0.22074954, -0.13758895, 0.2026092, 0.105174676, 0.11423842, 0.01239595, -0.12084066, 0.039877214, -0.22007395, -0.1703105, -0.121511586, 0.1487135, 0.13819724, -0.104532786, -0.0085047, 0.1507459, 0.23431942, 0.093546025, 0.03184169]
        // Weight Grad:  [-0.4959659, -1.9668058, -3.2469723, 1.080984, -2.0191822, -3.1055114, 2.659749, -1.03474, -2.5713987, 2.3330722, 0.36369538, -0.6405554, 3.7777996, 1.562422, 0.3727634, -2.3912354, -3.4810114, -2.5887141, -0.4959659, -1.9668058, -3.2469723, 1.080984, -2.0191822, -3.1055114, 2.659749, -1.03474, -2.5713987, 2.3330722, 0.36369538, -0.6405554, 3.7777996, 1.562422, 0.3727634, -2.3912354, -3.4810114, -2.5887141, -0.4959659, -1.9668058, -3.2469723, 1.080984, -2.0191822, -3.1055114, 2.659749, -1.03474, -2.5713987, 2.3330722, 0.36369538, -0.6405554, 3.7777996, 1.562422, 0.3727634, -2.3912354, -3.4810114, -2.5887141]
        // Bias:  [0.15803514, -0.13878204, 0.04392171]
        // Bias Grad:  [16.0, 16.0, 16.0]
        let input = vec![
            -0.45032975,
            -0.5730771,
            -0.5553584,
            0.59432304,
            1.5419426,
            0.5073344,
            -0.59103316,
            -0.5692481,
            0.18855357,
            -0.069072686,
            -0.49492535,
            -1.4959149,
            -0.19383712,
            0.44551218,
            1.3252748,
            -1.629326,
            2.0819554,
            1.7067116,
            2.3803675,
            -1.1256016,
            -0.3169981,
            -1.0924683,
            -0.0851943,
            -0.093348235,
            -0.7607072,
            -1.599082,
            0.018486667,
            -0.7504268,
            0.18540798,
            0.62113833,
            0.63818157,
            -0.24600095,
        ];
        let output = vec![
            -0.23461546,
            -0.7300762,
            -0.43270034,
            -0.27294904,
            -0.22774543,
            -0.165058,
            -0.018641159,
            0.3037408,
            0.639988,
            0.96370995,
            0.83733726,
            0.09503274,
            0.2800117,
            0.18681201,
            -0.1966653,
            -0.1629044,
            0.38531667,
            -0.44342816,
            0.3783964,
            -0.6913619,
            0.032867774,
            -0.6861952,
            -0.4665652,
            -0.46068507,
            -0.15072693,
            -0.5189232,
            -0.3886519,
            -0.25197878,
            0.075091064,
            0.38447255,
            0.032681137,
            0.43084383,
            -0.1430219,
            0.050474584,
            -0.48825723,
            -0.178395,
            0.08189282,
            0.32435876,
            0.248155,
            -0.03715375,
            -0.008421779,
            -0.29653978,
            0.3233031,
            0.47368968,
            -0.22343802,
            -0.29611385,
            0.045936823,
            -0.23833159,
        ];
        let grad = vec![
            -0.06312838,
            0.05240719,
            0.05240719,
            0.21505278,
            -0.07415994,
            0.063570745,
            0.063570745,
            0.22900042,
            -0.07415994,
            0.063570745,
            0.063570745,
            0.22900042,
            -0.0014246926,
            0.13951382,
            0.13951382,
            0.005797662,
            -0.73124456,
            -0.7982433,
            -0.7982433,
            -0.098860174,
            -0.57463914,
            -0.689119,
            -0.689119,
            -0.12428501,
            -0.57463914,
            -0.689119,
            -0.689119,
            -0.12428501,
            -0.22594097,
            -0.37261552,
            -0.37261552,
            -0.085577406,
        ];
        let weight = vec![
            -0.0017646605,
            0.12644097,
            -0.1939936,
            -0.1734625,
            -0.090781756,
            0.063205294,
            -0.0046700113,
            0.18688585,
            -0.020917172,
            0.06236978,
            -0.071232304,
            -0.046330906,
            -0.2251778,
            -0.15610139,
            -0.09716192,
            0.008731253,
            0.0931814,
            0.14142673,
            -0.15979224,
            -0.10263957,
            0.0856111,
            0.19572432,
            -0.048507567,
            0.17637877,
            -0.03799128,
            0.024940623,
            0.21342279,
            -0.218654,
            -0.14838351,
            -0.05967162,
            -0.09187673,
            0.20364694,
            -0.1527774,
            -0.1085015,
            -0.16467114,
            -0.22074954,
            -0.13758895,
            0.2026092,
            0.105174676,
            0.11423842,
            0.01239595,
            -0.12084066,
            0.039877214,
            -0.22007395,
            -0.1703105,
            -0.121511586,
            0.1487135,
            0.13819724,
            -0.104532786,
            -0.0085047,
            0.1507459,
            0.23431942,
            0.093546025,
            0.03184169,
        ];
        let weight_grad = vec![
            -0.4959659, -1.9668058, -3.2469723, 1.080984, -2.0191822, -3.1055114, 2.659749,
            -1.03474, -2.5713987, 2.3330722, 0.36369538, -0.6405554, 3.7777996, 1.562422,
            0.3727634, -2.3912354, -3.4810114, -2.5887141, -0.4959659, -1.9668058, -3.2469723,
            1.080984, -2.0191822, -3.1055114, 2.659749, -1.03474, -2.5713987, 2.3330722,
            0.36369538, -0.6405554, 3.7777996, 1.562422, 0.3727634, -2.3912354, -3.4810114,
            -2.5887141, -0.4959659, -1.9668058, -3.2469723, 1.080984, -2.0191822, -3.1055114,
            2.659749, -1.03474, -2.5713987, 2.3330722, 0.36369538, -0.6405554, 3.7777996, 1.562422,
            0.3727634, -2.3912354, -3.4810114, -2.5887141,
        ];
        let bias = vec![0.15803514, -0.13878204, 0.04392171];
        let bias_grad = vec![16.0, 16.0, 16.0];
        let input = from_vec(input, [1, 2, 4, 4]);

        let output = Matrix::<_, DimDyn, _>::from_vec(output, [1, 3, 4, 4]);
        let grad = Matrix::<_, DimDyn, _>::from_vec(grad, [1, 2, 4, 4]);
        let weight = from_vec(weight, [3, 2, 3, 3]);
        let weight_grad = Matrix::<_, DimDyn, _>::from_vec(weight_grad, [3, 2, 3, 3]);
        let bias = from_vec(bias, [1, 3, 1, 1]);
        let bias_grad = Matrix::<_, DimDyn, _>::from_vec(bias_grad, [1, 3, 1, 1]);

        let pred = conv2d::<f32, D>(
            input.clone(),
            weight.clone(),
            (1, 1),
            (1, 1),
            Some(bias.clone()),
            None,
        );
        pred.backward();

        assert_val_eq!(pred.clone(), output, 2e-6);
        assert_val_eq_grad!(input, grad, 2e-5);

        assert_val_eq_grad!(weight, weight_grad, 2e-5);
        assert_val_eq_grad!(bias, bias_grad, 2e-6);
    }
    run_test!(
        conv2d_with_bias,
        conv2d_with_bias_cpu,
        conv2d_with_bias_nvidia
    );
}

use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::DimDyn,
    nn::batch_norm::{
        try_batch_norm_2d_backward, try_batch_norm_2d_forward_inference,
        try_batch_norm_2d_forward_trian, BatchNorm2dBackwardConfig, BatchNorm2dConfig,
        BatchNorm2dInferenceConfig,
    },
    num::Num,
};

use crate::{creator::zeros::zeros_like, is_train, Function, Variable, VariableWeak};

#[derive(Default)]
pub struct BatchNorm2dInner<T: Num> {
    pub train: Option<BatchNorm2dConfig<T>>,
    pub inference: Option<BatchNorm2dInferenceConfig<T>>,
    pub bckwd: Option<BatchNorm2dBackwardConfig<T>>,
}

impl<T: Num> BatchNorm2dInner<T> {
    pub fn new(dim: DimDyn) -> Self {
        let train = BatchNorm2dConfig::<T>::new(dim);
        let inference = BatchNorm2dInferenceConfig::<T>::new(dim);
        let bckwd = BatchNorm2dBackwardConfig::<T>::new(dim);
        Self {
            train: Some(train),
            inference: Some(inference),
            bckwd: Some(bckwd),
        }
    }
}

#[derive(Clone, Default)]
pub struct BatchNorm2dAutoGradConfig<T: Num> {
    inner: RefCell<Rc<BatchNorm2dInner<T>>>,
    dim: RefCell<DimDyn>,
}

impl<T: Num> BatchNorm2dAutoGradConfig<T> {
    pub fn new(dim: &[usize]) -> Self {
        let dim = DimDyn::from(dim);
        let inner = Rc::new(BatchNorm2dInner::new(dim));
        Self {
            inner: RefCell::new(inner),
            dim: RefCell::new(dim),
        }
    }

    pub fn update_shape(&self, dim: &[usize]) {
        let dim = DimDyn::from(dim);
        *self.inner.borrow_mut() = Rc::new(BatchNorm2dInner::new(dim));
        self.dim.replace(dim);
    }

    pub fn get_shape(&self) -> DimDyn {
        *self.dim.borrow()
    }
}

struct BatchNorm2d<T: Num, D: Device> {
    momentum: f64,
    x: Variable<T, D>,
    y: VariableWeak<T, D>,
    scale: Variable<T, D>,
    bias: Variable<T, D>,
    mean: Variable<T, D>,
    variance: Variable<T, D>,
    saving_mean: Variable<T, D>,
    saving_inv_variance: Variable<T, D>,
    config: BatchNorm2dAutoGradConfig<T>,
}

struct BatchNorm2dBkwd<T: Num, D: Device> {
    x: Variable<T, D>,
    y_grad: Variable<T, D>,
    x_grad: VariableWeak<T, D>,
    scale: Variable<T, D>,
    scale_grad: VariableWeak<T, D>,
    bias_grad: VariableWeak<T, D>,
    saving_mean: Variable<T, D>,
    saving_inv_variance: Variable<T, D>,
    config: BatchNorm2dAutoGradConfig<T>,
}

impl<T: Num, D: Device> Function<T, D> for BatchNorm2d<T, D> {
    fn forward(&self) {
        if is_train() {
            let config_train = &self.config.inner.borrow().train;
            try_batch_norm_2d_forward_trian(
                self.momentum,
                self.x.get_data().to_ref(),
                self.y.upgrade().unwrap().get_data_mut().to_ref_mut(),
                self.scale.get_data().to_ref(),
                self.bias.get_data().to_ref(),
                self.mean.get_data_mut().to_ref_mut(),
                self.variance.get_data_mut().to_ref_mut(),
                Some(self.saving_mean.get_data_mut().to_ref_mut()),
                Some(self.saving_inv_variance.get_data_mut().to_ref_mut()),
                config_train,
            )
            .unwrap();
        } else {
            let config_inference = &self.config.inner.borrow().inference;
            try_batch_norm_2d_forward_inference(
                self.x.get_data().to_ref(),
                self.y.upgrade().unwrap().get_data_mut().to_ref_mut(),
                self.scale.get_data().to_ref(),
                self.bias.get_data().to_ref(),
                self.mean.get_data().to_ref(),
                self.variance.get_data().to_ref(),
                config_inference,
            )
            .unwrap();
        };
    }

    fn backward(&self) {
        let grads = batch_norm_2d_bkwd(
            self.x.clone(),
            self.y.upgrade().unwrap().get_grad().unwrap(),
            self.scale.clone(),
            self.saving_mean.clone(),
            self.saving_inv_variance.clone(),
            self.config.clone(),
        );

        self.x.set_grad(grads.x_grad);
        self.scale.set_grad(grads.scale_grad);
        self.bias.set_grad(grads.bias_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![
            self.x.clone(),
            self.scale.clone(),
            self.bias.clone(),
            self.mean.clone(),
            self.variance.clone(),
        ]
    }
}

impl<T: Num, D: Device> Function<T, D> for BatchNorm2dBkwd<T, D> {
    fn forward(&self) {
        let config_bkwd = &self.config.inner.borrow().bckwd;
        try_batch_norm_2d_backward(
            self.x.get_data().to_ref(),
            self.y_grad.get_data().to_ref(),
            self.x_grad.upgrade().unwrap().get_data_mut().to_ref_mut(),
            self.scale.get_data().to_ref(),
            self.scale_grad
                .upgrade()
                .unwrap()
                .get_data_mut()
                .to_ref_mut(),
            self.bias_grad
                .upgrade()
                .unwrap()
                .get_data_mut()
                .to_ref_mut(),
            Some(self.saving_mean.get_data().to_ref()),
            Some(self.saving_inv_variance.get_data().to_ref()),
            config_bkwd,
        )
        .unwrap();
    }

    fn backward(&self) {
        panic!(
            r"BatchNorm2d's backward of backward is not implemented. 
            if you want to continue, please implement it."
        );
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.scale.clone(), self.y_grad.clone()]
    }
}

pub fn batch_norm_2d<T: Num, D: Device>(
    x: Variable<T, D>,
    scale: Variable<T, D>,
    bias: Variable<T, D>,
    mean: Variable<T, D>,
    variance: Variable<T, D>,
    momentum: f64,
    config: BatchNorm2dAutoGradConfig<T>,
) -> Variable<T, D> {
    let y = zeros_like(&x);
    let saving_mean = zeros_like(&mean);
    let saving_inv_variance = zeros_like(&variance);
    let batch_norm = BatchNorm2d {
        momentum,
        x,
        y: y.clone().downgrade(),
        scale,
        bias,
        mean,
        variance,
        saving_mean,
        saving_inv_variance,
        config,
    };
    batch_norm.forward();
    y.set_creator(Rc::new(RefCell::new(Box::new(batch_norm))));
    y
}

struct BatchNorm2dGradOut<T: Num, D: Device> {
    x_grad: Variable<T, D>,
    scale_grad: Variable<T, D>,
    bias_grad: Variable<T, D>,
}

fn batch_norm_2d_bkwd<T: Num, D: Device>(
    x: Variable<T, D>,
    y_grad: Variable<T, D>,
    scale: Variable<T, D>,
    saving_mean: Variable<T, D>,
    saving_inv_variance: Variable<T, D>,
    config: BatchNorm2dAutoGradConfig<T>,
) -> BatchNorm2dGradOut<T, D> {
    let x_grad = zeros_like(&x);
    let scale_grad = zeros_like(&scale);
    let bias_grad = zeros_like(&scale);
    let batch_norm_bkwd = BatchNorm2dBkwd {
        x,
        y_grad,
        x_grad: x_grad.clone().downgrade(),
        scale,
        scale_grad: scale_grad.clone().downgrade(),
        bias_grad: bias_grad.clone().downgrade(),
        saving_mean,
        saving_inv_variance,
        config,
    };

    batch_norm_bkwd.forward();
    let function: Rc<RefCell<Box<dyn Function<T, D>>>> =
        Rc::new(RefCell::new(Box::new(batch_norm_bkwd)));
    x_grad.set_creator(function.clone());
    scale_grad.set_creator(function.clone());
    bias_grad.set_creator(function);
    BatchNorm2dGradOut {
        x_grad,
        scale_grad,
        bias_grad,
    }
}

#[cfg(test)]
mod batch_norm_2d {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::Variable;

    use super::{batch_norm_2d, BatchNorm2dAutoGradConfig};

    struct TestConfig<D: Device> {
        x: Matrix<Owned<f32>, DimDyn, D>,
        x_grad: Matrix<Owned<f32>, DimDyn, D>,
        y: Matrix<Owned<f32>, DimDyn, D>,
        y_grad: Matrix<Owned<f32>, DimDyn, D>,
        scale: Matrix<Owned<f32>, DimDyn, D>,
        bias: Matrix<Owned<f32>, DimDyn, D>,
        mean: Matrix<Owned<f32>, DimDyn, D>,
        variance: Matrix<Owned<f32>, DimDyn, D>,
        scale_grad: Matrix<Owned<f32>, DimDyn, D>,
        bias_grad: Matrix<Owned<f32>, DimDyn, D>,
        momentum: f64,
        config: BatchNorm2dAutoGradConfig<f32>,
    }

    fn small_test<D: Device>() {
        let test_case = small_test_case::<D>();

        let x = Variable::new(test_case.x.to_ref());
        let scale = Variable::new(test_case.scale.to_ref());
        let bias = Variable::new(test_case.bias.to_ref());
        let mean = Variable::new(test_case.mean.to_ref());
        let variance = Variable::new(test_case.variance.to_ref());

        let y = batch_norm_2d(
            x.clone(),
            scale.clone(),
            bias.clone(),
            mean,
            variance,
            test_case.momentum,
            test_case.config,
        );

        assert_val_eq!(y.clone(), test_case.y, 6e-4);
        let hoge = y * Variable::new(test_case.y_grad.to_ref());
        hoge.backward();
        assert_val_eq_grad!(x, test_case.x_grad, 7e-4);
        assert_val_eq_grad!(scale, test_case.scale_grad, 6e-4);
        assert_val_eq_grad!(bias, test_case.bias_grad, 6e-4);
    }
    run_test!(small_test, small_test_cpu, small_test_gpu);

    fn small_test_case<D: Device>() -> TestConfig<D> {
        let x = [
            -1.1258398,
            -1.1523602,
            -0.25057858,
            -0.43387884,
            0.84871036,
            0.6920092,
            -0.31601277,
            -2.1152196,
            0.32227492,
            -1.2633348,
            0.3499832,
            0.3081339,
            0.11984151,
            1.2376579,
            1.1167772,
            -0.24727765,
            -1.3526537,
            -1.6959313,
            0.5666505,
            0.7935084,
            0.59883946,
            -1.5550951,
            -0.3413603,
            1.8530061,
            0.7501894,
            -0.58549714,
            -0.17339702,
            0.18347792,
            1.3893661,
            1.5863343,
            0.94629836,
            -0.8436768,
            -0.6135831,
            0.03159274,
            -0.49267703,
            0.24841475,
            0.43969586,
            0.112411186,
            0.64079237,
            0.44115627,
            -0.10230965,
            0.792444,
            -0.28966758,
            0.052507486,
            0.5228604,
            2.3022053,
            -1.4688939,
            -1.5866888,
            -0.6730899,
            0.8728312,
            1.0553576,
            0.17784412,
            -0.23033547,
            -0.3917544,
            0.54329467,
            -0.39515752,
            -0.4462172,
            0.7440207,
            1.5209795,
            3.4105027,
            -1.5311843,
            -1.234135,
            1.8197253,
            -0.5515287,
            -0.569248,
            0.9199713,
            1.1108161,
            1.2898738,
            -1.4781743,
            2.5672328,
            -0.4731198,
            0.33555073,
            -1.629326,
            -0.54974365,
            -0.47983426,
            -0.49968216,
            -1.06698,
            1.1149396,
            -0.14067143,
            0.80575365,
            -0.09334823,
            0.6870502,
            -0.83831537,
            0.00089182175,
            0.8418941,
            -0.40003455,
            1.039462,
            0.3581531,
            -0.24600095,
            2.3025165,
            -1.8816892,
            -0.049727023,
            -1.0449786,
            -0.95650053,
            0.03353186,
            0.7100866,
        ];
        let bias = [2.0575912, -0.03542188, 0.06271883];
        let weight = [-0.7663063, 1.0992506, 2.7565384];
        let prev_mean = [0.16644886, -0.8278146, -1.3502742];
        let prev_var = [-0.47117928, 1.3144886, 0.12074463];
        let y = [
            2.906542,
            2.9251065,
            2.293855,
            2.422166,
            1.5243473,
            1.6340389,
            2.3396592,
            3.5991127,
            1.8928547,
            3.0027893,
            1.8734589,
            1.9027535,
            2.034559,
            1.2520821,
            1.3366992,
            2.2915442,
            -1.5002401,
            -1.8450762,
            0.42777777,
            0.6556656,
            0.4601128,
            -1.7036005,
            -0.4843554,
            1.7199733,
            0.6121499,
            -0.7296006,
            -0.3156296,
            0.04286556,
            1.2542285,
            1.4520909,
            0.8091492,
            -0.9889524,
            -1.9158504,
            0.017554268,
            -1.55353,
            0.6673069,
            1.2405207,
            0.25974366,
            1.8431487,
            1.2448971,
            -0.38371232,
            2.2976043,
            -0.9451696,
            0.080229685,
            1.4897408,
            6.821921,
            -4.478968,
            -4.8319654,
            2.5896149,
            1.5074626,
            1.3796933,
            1.993957,
            2.2796848,
            2.3926787,
            1.7381399,
            2.395061,
            2.430803,
            1.5976306,
            1.0537556,
            -0.2689197,
            3.1902852,
            2.9823494,
            0.84463215,
            2.5045216,
            -0.7132777,
            0.78270257,
            0.9744138,
            1.1542845,
            -1.6263306,
            2.4374425,
            -0.6167131,
            0.19562875,
            -1.7781684,
            -0.6936848,
            -0.623458,
            -0.643396,
            -1.2132695,
            0.978556,
            -0.28275543,
            0.6679664,
            -0.35685754,
            1.98177,
            -2.5893078,
            -0.074447475,
            2.445792,
            -1.2759073,
            3.0378456,
            0.9961608,
            -0.8143134,
            6.822853,
            -5.715996,
            -0.22613744,
            -3.2086174,
            -2.9434743,
            0.023365237,
            2.0508032,
        ];
        let x_grad = [
            0.28300184,
            -0.93610215,
            0.29200143,
            0.7483416,
            -0.098302774,
            0.41916126,
            -0.56200033,
            0.21534747,
            -1.0423822,
            -0.6819873,
            -0.41098782,
            0.63031906,
            -0.22403057,
            -0.074631736,
            0.5188781,
            -0.30130252,
            0.41519412,
            -0.9002232,
            0.44782448,
            -2.0863774,
            -0.3029541,
            0.5983221,
            2.1732514,
            1.002588,
            0.13616884,
            0.5537893,
            -1.6048535,
            -0.33637834,
            2.6140049,
            1.1925341,
            0.3480426,
            -0.08439568,
            -2.0070481,
            -3.3533998,
            5.425738,
            1.5095177,
            -3.7216716,
            -0.5247467,
            0.798083,
            0.1575643,
            2.5028248,
            4.308768,
            2.3008642,
            1.1123875,
            3.7429547,
            2.1617568,
            2.9372573,
            -2.4336362,
            0.04158987,
            0.10034412,
            -1.3560778,
            -0.5757772,
            1.4964242,
            0.69442236,
            -0.1982,
            0.6807664,
            -0.22737995,
            -0.8158201,
            1.0029143,
            -0.13899583,
            0.3256538,
            -0.8646269,
            0.22587873,
            0.8335607,
            0.44324234,
            -1.6979185,
            0.43024784,
            0.04143095,
            -0.6486931,
            -0.64391875,
            0.10925212,
            -2.2644544,
            0.52335936,
            -0.7549458,
            0.25733638,
            0.46163946,
            -0.24749541,
            -1.6739483,
            1.5959818,
            -0.097653344,
            -0.7424374,
            -3.8802469,
            -1.1844432,
            -0.5777171,
            -0.55032843,
            -3.7924376,
            -1.1667422,
            -5.169531,
            -2.5999074,
            0.062334046,
            -0.24346678,
            1.8333617,
            -0.5688741,
            4.110656,
            0.92301893,
            -1.3704543,
        ];
        let bias_grad = [-4.602761, 12.493873, -9.31948];
        let weight_grad = [4.430987, -8.083071, -5.8678145];
        let y_grad = [
            -0.70152366,
            1.0366868,
            -0.6036701,
            -1.2787654,
            0.092950225,
            -0.6660997,
            0.6080472,
            -0.73001987,
            1.3750379,
            0.6596311,
            0.47655705,
            -1.0163072,
            0.18036698,
            0.10833187,
            -0.75482327,
            0.24431853,
            1.1403507,
            -0.08988206,
            0.72979575,
            -1.845319,
            -0.02501994,
            1.3693811,
            2.6570232,
            0.9851194,
            0.37718192,
            1.1012348,
            -1.1427782,
            0.037585836,
            2.6962764,
            1.2357637,
            0.5428298,
            0.52553034,
            -0.82936686,
            -1.4072566,
            1.6268467,
            0.1722732,
            -1.6115024,
            -0.47944778,
            -0.14335093,
            -0.31729499,
            0.5736546,
            0.9979312,
            0.54360944,
            0.07880439,
            0.8628601,
            -0.019489521,
            0.99104714,
            -0.7777345,
            -0.29938444,
            -0.18777807,
            1.9158976,
            0.69019526,
            -2.3217015,
            -1.1964102,
            0.19702816,
            -1.1773323,
            0.113552175,
            1.1047257,
            -1.395172,
            0.4751187,
            -0.8137258,
            0.92423636,
            -0.24734129,
            -1.4153874,
            0.9874366,
            -1.4878075,
            0.5866875,
            0.15829548,
            0.11024585,
            -0.8188127,
            0.63276637,
            -1.9168797,
            1.311892,
            -0.20983858,
            0.7817313,
            0.9896925,
            0.41471335,
            -1.5089506,
            2.036037,
            0.13159046,
            -0.51107377,
            -1.7137278,
            -0.5100648,
            -0.47489306,
            -0.6334037,
            -1.4677203,
            -0.87848496,
            -2.0783968,
            -1.1004796,
            -0.7201275,
            0.011930629,
            0.33977303,
            -0.26345232,
            1.280466,
            0.019394945,
            -0.8808039,
        ];
        let x = Matrix::<Owned<f32>, DimDyn, D>::from_vec(x.to_vec(), &[2, 3, 4, 4]);
        let y = Matrix::<Owned<f32>, DimDyn, D>::from_vec(y.to_vec(), &[2, 3, 4, 4]);
        let x_grad = Matrix::<Owned<f32>, DimDyn, D>::from_vec(x_grad.to_vec(), &[2, 3, 4, 4]);
        let y_grad = Matrix::<Owned<f32>, DimDyn, D>::from_vec(y_grad.to_vec(), &[2, 3, 4, 4]);
        let scale = Matrix::<Owned<f32>, DimDyn, D>::from_vec(weight.to_vec(), &[3]);
        let bias = Matrix::<Owned<f32>, DimDyn, D>::from_vec(bias.to_vec(), &[3]);
        let mean = Matrix::<Owned<f32>, DimDyn, D>::from_vec(prev_mean.to_vec(), &[3]);
        let variance = Matrix::<Owned<f32>, DimDyn, D>::from_vec(prev_var.to_vec(), &[3]);
        let scale_grad = Matrix::<Owned<f32>, DimDyn, D>::from_vec(weight_grad.to_vec(), &[3]);
        let bias_grad = Matrix::<Owned<f32>, DimDyn, D>::from_vec(bias_grad.to_vec(), &[3]);
        let config = BatchNorm2dAutoGradConfig::new(&[2, 3, 4, 4]);
        TestConfig {
            x,
            x_grad,
            y,
            y_grad,
            scale,
            bias,
            mean,
            variance,
            scale_grad,
            bias_grad,
            momentum: 0.1,
            config,
        }
    }
}

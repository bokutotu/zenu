use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    constructor::{ones::Ones, zeros::Zeros},
    dim::DimTrait,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::OwnedMatrixDyn,
    num::Num,
    operation::{
        basic_operations::{MatrixAddAssign, MatrixMulAssign, MatrixSqrt},
        copy_from::CopyFrom,
        mean::Mean,
        reshape::Reshape,
        transpose::TransposeInplace,
        var::Variance,
    },
};

use crate::{creator::zeros::zeros, is_train, Function, Variable, VariableWeak};

use super::{reshape::reshape, sum::sum, transpose::transpose_by_index};

struct BatchNorm<T: Num> {
    mean: Variable<T>,
    variance: Variable<T>,
    decay: Variable<T>,
    epsilon: Variable<T>,
    inv_std: Variable<T>,
    gamma: Variable<T>,
    beta: Variable<T>,
    input: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> BatchNorm<T> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        mean: Variable<T>,
        variance: Variable<T>,
        decay: Variable<T>,
        epsilon: Variable<T>,
        inv_std: Variable<T>,
        gamma: Variable<T>,
        beta: Variable<T>,
        input: Variable<T>,
        output: Variable<T>,
    ) -> Self {
        let output = output.downgrade();
        Self {
            mean,
            variance,
            decay,
            epsilon,
            inv_std,
            gamma,
            beta,
            input,
            output,
        }
    }
}

impl<T: Num> Function<T> for BatchNorm<T> {
    fn forward(&self) {
        let input_mat = self.input.get_data();
        let input_shape = input_mat.shape();
        let num_channels = input_shape.len();
        let input_mat = if num_channels == 4 {
            let channel = input_mat.shape()[1];
            let num_elm = input_mat.shape().num_elm();
            let input_mat = input_mat.transepose_by_index(&[0, 2, 3, 1]);
            input_mat.reshape_new_matrix([num_elm / channel, channel])
        } else {
            input_mat
        };

        let xc;

        if is_train() {
            let mean = input_mat.mean(Some(0), false);
            let var = input_mat.variance(Some(0), false);
            let var_eps = var.to_view() + self.epsilon.get_data();
            let mut zeros = OwnedMatrixDyn::zeros_like(var_eps.to_view());

            zeros.to_view_mut().sqrt(var_eps);

            let inv_std = OwnedMatrixDyn::ones(var.shape()) / zeros;
            xc = (input_mat.to_view() - mean.to_view()) * inv_std.to_view();
            let m = input_mat.shape().num_elm() / self.gamma.get_data().shape().num_elm();
            let s = if m - 1 > 1 { m - 1 } else { 1 };
            let adjust = m / s;

            self.mean.get_data_mut().mul_assign(self.decay.get_data());
            self.mean.get_data_mut().to_view_mut().add_assign(
                mean.to_view()
                    * (OwnedMatrixDyn::ones(self.decay.get_data().shape()) - self.decay.get_data()),
            );

            self.variance
                .get_data_mut()
                .mul_assign(self.decay.get_data());

            self.variance.get_data_mut().to_view_mut().add_assign(
                var.to_view()
                    * (OwnedMatrixDyn::ones(self.decay.get_data().shape()) - self.decay.get_data())
                    * T::from_usize(adjust),
            );

            self.inv_std
                .get_data_mut()
                .to_view_mut()
                .copy_from(&inv_std.to_view());
        } else {
            let var_eps = self.variance.get_data().to_view() + self.epsilon.get_data();
            let mut zeros = OwnedMatrixDyn::zeros_like(var_eps.to_view());
            zeros.to_view_mut().sqrt(var_eps);
            let inv_std = OwnedMatrixDyn::ones(self.variance.get_data().shape()) / zeros;
            xc = (input_mat.to_view() - self.mean.get_data().to_view()) * inv_std.to_view();
        }

        let output =
            self.gamma.get_data().to_view() * xc.to_view() + self.beta.get_data().to_view();

        let output = if num_channels == 4 {
            let output = output.reshape_new_matrix([
                input_shape[0],
                input_shape[2],
                input_shape[3],
                input_shape[1],
            ]);
            output.transpose_by_index_inplace(&[0, 3, 1, 2])
        } else {
            output
        };

        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_view_mut()
            .copy_from(&output);
    }

    fn backward(&self) {
        if !is_train() {
            panic!("backward is called in inference mode");
        }
        let input_shape = self.input.get_data().shape();
        let shape_len = input_shape.len();
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        let output_grad = if shape_len == 4 {
            let channel = output_grad.get_data().shape()[1];
            let num_elm = output_grad.get_data().shape().num_elm();
            let output_grad = transpose_by_index(output_grad, vec![0, 2, 3, 1]);
            reshape(output_grad, &[num_elm / channel, channel])
        } else {
            output_grad
        };
        let batch_size = output_grad.get_data().shape()[0];
        let input = if self.input.get_data().shape().len() == 4 {
            let channel = self.input.get_data().shape()[1];
            let num_elm = self.input.get_data().shape().num_elm();
            let input = transpose_by_index(self.input.clone(), vec![0, 2, 3, 1]);
            reshape(input, &[num_elm / channel, channel])
        } else {
            self.input.clone()
        };
        let mean = sum(input.clone(), 0, false) / Variable::from(T::from_usize(batch_size));
        let xc = (input - mean) * self.inv_std.clone();
        let beta_grad = sum(output_grad.clone(), 0, false);
        let gamma_grad = sum(xc.clone() * output_grad.clone(), 0, false);

        let beta_grad_batch_size = beta_grad.clone() / Variable::from(T::from_usize(batch_size));
        let xc_gamma_grad_batch_size =
            xc.clone() * gamma_grad.clone() / Variable::from(T::from_usize(batch_size));
        let input_grad = output_grad.clone() - beta_grad_batch_size - xc_gamma_grad_batch_size;
        let input_grad = input_grad * self.gamma.clone() * self.inv_std.clone();

        if shape_len == 4 {
            let input_grad = reshape(
                input_grad,
                &[
                    input_shape[0],
                    input_shape[2],
                    input_shape[3],
                    input_shape[1],
                ],
            );
            let input_grad = transpose_by_index(input_grad, vec![0, 3, 1, 2]);
            self.input.set_grad(input_grad);
        } else {
            self.input.set_grad(input_grad);
        }
        self.gamma.set_grad(gamma_grad);
        self.beta.set_grad(beta_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![
            self.mean.clone(),
            self.variance.clone(),
            self.decay.clone(),
            self.epsilon.clone(),
            self.inv_std.clone(),
            self.input.clone(),
        ]
    }
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm<T: Num>(
    mean: Variable<T>,
    variance: Variable<T>,
    decay: Variable<T>,
    epsilon: Variable<T>,
    gamma: Variable<T>,
    beta: Variable<T>,
    input: Variable<T>,
) -> Variable<T> {
    let output_shape = input.get_data().shape();
    let output = zeros(output_shape);
    let inv_std = zeros(variance.get_data().shape());
    let batch_norm = BatchNorm::new(
        mean,
        variance,
        decay,
        epsilon,
        inv_std,
        gamma,
        beta,
        input,
        output.clone(),
    );
    batch_norm.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(batch_norm))));
    output
}

#[cfg(test)]
mod batch_norm {
    use crate::{no_train, set_train};
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use super::batch_norm;

    #[test]
    fn batch_norm_medium() {
        let input = (1..31).map(|x| x as f64).collect::<Vec<f64>>();
        let input = OwnedMatrixDyn::from_vec(input, &[10, 3]);
        let mean = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let var = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let decay = OwnedMatrixDyn::from_vec(vec![0.9], &[]);
        let epsilon = OwnedMatrixDyn::from_vec(vec![1e-5], &[]);
        let gamma = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let delta = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let input = crate::Variable::from(input);
        let mean = crate::Variable::from(mean);
        let var = crate::Variable::from(var);
        let decay = crate::Variable::from(decay);
        let epsilon = crate::Variable::from(epsilon);
        let gamma = crate::Variable::from(gamma);
        let beta = crate::Variable::from(delta);
        let output = batch_norm(
            mean,
            var,
            decay,
            epsilon,
            gamma.clone(),
            beta.clone(),
            input.clone(),
        );
        output.backward();
        let ans = vec![
            -0.5666988,
            -1.1333976,
            -1.70009639,
            -0.21854351,
            -0.43708702,
            -0.65563053,
            0.12961178,
            0.25922356,
            0.38883534,
            0.47776707,
            0.95553413,
            1.4333012,
            0.82592236,
            1.65184471,
            2.47776707,
            1.17407764,
            2.34815529,
            3.52223293,
            1.52223293,
            3.04446587,
            4.5666988,
            1.87038822,
            3.74077644,
            5.61116466,
            2.21854351,
            4.43708702,
            6.65563053,
            2.5666988,
            5.1333976,
            7.70009639,
        ];
        let ans = OwnedMatrixDyn::from_vec(ans, &[10, 3]);
        assert!((ans - output.get_data()).asum() < 1e-6);
        let input_grad_ans = vec![
            4.037174091273418e-18,
            8.074348182546835e-18,
            1.2111522273820252e-17,
            3.1400242932126583e-18,
            6.2800485864253165e-18,
            9.420072879637973e-18,
            2.2428744951518985e-18,
            4.485748990303797e-18,
            6.728623485455695e-18,
            1.3457246970911395e-18,
            2.691449394182279e-18,
            4.037174091273418e-18,
            4.485748990303797e-19,
            8.971497980607594e-19,
            1.345724697091139e-18,
            -4.485748990303797e-19,
            -8.971497980607594e-19,
            -1.345724697091139e-18,
            -1.3457246970911395e-18,
            -2.691449394182279e-18,
            -4.037174091273418e-18,
            -2.2428744951518985e-18,
            -4.485748990303797e-18,
            -6.728623485455695e-18,
            -3.1400242932126583e-18,
            -6.2800485864253165e-18,
            -9.420072879637973e-18,
            -4.037174091273418e-18,
            -8.074348182546835e-18,
            -1.2111522273820252e-17,
        ];
        let input_grad_ans = OwnedMatrixDyn::from_vec(input_grad_ans, &[10, 3]);
        let gamma_grad = vec![
            2.220446049250313e-16,
            2.220446049250313e-16,
            2.220446049250313e-16,
        ];
        let gamma_grad = OwnedMatrixDyn::from_vec(gamma_grad, &[3]);
        let beta_grad = vec![10., 10., 10.];
        let beta_grad = OwnedMatrixDyn::from_vec(beta_grad, &[3]);
        assert!((input_grad_ans - input.get_grad().unwrap().get_data()).asum() < 1e-25);
        assert!((gamma_grad - gamma.get_grad().unwrap().get_data()).asum() < 1e-25);
        assert!((beta_grad - beta.get_grad().unwrap().get_data()).asum() < 1e-25);
    }

    #[test]
    fn batch_norm_medium_no_train() {
        no_train();
        let input = (1..31).map(|x| x as f64).collect::<Vec<f64>>();
        let input = OwnedMatrixDyn::from_vec(input, &[10, 3]);
        let mean = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let var = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let decay = OwnedMatrixDyn::from_vec(vec![0.9], &[]);
        let epsilon = OwnedMatrixDyn::from_vec(vec![1e-5], &[]);
        let gamma = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let delta = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let input = crate::Variable::from(input);
        let mean = crate::Variable::from(mean);
        let var = crate::Variable::from(var);
        let decay = crate::Variable::from(decay);
        let epsilon = crate::Variable::from(epsilon);
        let gamma = crate::Variable::from(gamma);
        let beta = crate::Variable::from(delta);
        let output = batch_norm(
            mean,
            var,
            decay,
            epsilon,
            gamma.clone(),
            beta.clone(),
            input.clone(),
        );
        let output = output.get_data();
        let ans = vec![
            1.0,
            2.0,
            3.0,
            3.999985000112499,
            6.242630080557342,
            8.196143762474245,
            6.999970000224998,
            10.485260161114685,
            13.39228752494849,
            9.999955000337497,
            14.727890241672027,
            18.588431287422736,
            12.999940000449996,
            18.97052032222937,
            23.78457504989698,
            15.999925000562497,
            23.21315040278671,
            28.980718812371222,
            18.999910000674994,
            27.455780483344054,
            34.17686257484547,
            21.999895000787493,
            31.698410563901394,
            39.37300633731971,
            24.999880000899992,
            35.94104064445874,
            44.56915009979396,
            27.999865001012495,
            40.18367072501608,
            49.765293862268194,
        ];
        let ans = OwnedMatrixDyn::from_vec(ans, &[10, 3]);
        assert!((ans - output).asum() < 1e-6);
        set_train();
    }

    #[test]
    fn batch_norom_meduium_4d() {
        let input = (1..(10 * 3 * 3 * 3 + 1))
            .map(|x| x as f64)
            .collect::<Vec<f64>>();
        let input = OwnedMatrixDyn::from_vec(input, &[10, 3, 3, 3]);
        let mean = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let var = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let decay = OwnedMatrixDyn::from_vec(vec![0.9], &[]);
        let epsilon = OwnedMatrixDyn::from_vec(vec![1e-5], &[]);
        let gamma = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let delta = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let input = crate::Variable::from(input);
        let mean = crate::Variable::from(mean);
        let var = crate::Variable::from(var);
        let decay = crate::Variable::from(decay);
        let epsilon = crate::Variable::from(epsilon);
        let gamma = crate::Variable::from(gamma);
        let beta = crate::Variable::from(delta);
        let output = batch_norm(
            mean,
            var,
            decay,
            epsilon,
            gamma.clone(),
            beta.clone(),
            input.clone(),
        );
        output.backward();
        let output_data = output.get_data();
        let ans = vec![
            -0.6173812990842029,
            -0.6044937986930936,
            -0.5916062983019845,
            -0.5787187979108754,
            -0.5658312975197661,
            -0.552943797128657,
            -0.5400562967375477,
            -0.5271687963464387,
            -0.5142812959553293,
            -1.2347625981684058,
            -1.2089875973861872,
            -1.183212596603969,
            -1.1574375958217509,
            -1.1316625950395323,
            -1.105887594257314,
            -1.0801125934750955,
            -1.0543375926928773,
            -1.0285625919106587,
            -1.852143897252609,
            -1.8134813960792808,
            -1.7748188949059536,
            -1.7361563937326263,
            -1.6974938925592982,
            -1.658831391385971,
            -1.6201688902126428,
            -1.5815063890393155,
            -1.5428438878659883,
            -0.26941878852425494,
            -0.25653128813314563,
            -0.24364378774203654,
            -0.23075628735092724,
            -0.21786878695981815,
            -0.20498128656870884,
            -0.19209378617759976,
            -0.17920628578649045,
            -0.16631878539538136,
            -0.5388375770485099,
            -0.5130625762662913,
            -0.4872875754840731,
            -0.46151257470185447,
            -0.4357375739196363,
            -0.4099625731374177,
            -0.3841875723551995,
            -0.3584125715729809,
            -0.3326375707907627,
            -0.808256365572765,
            -0.7695938643994369,
            -0.7309313632261096,
            -0.6922688620527815,
            -0.6536063608794542,
            -0.6149438597061265,
            -0.5762813585327993,
            -0.5376188573594716,
            -0.4989563561861443,
            0.07854372203569315,
            0.09143122242680235,
            0.10431872281791155,
            0.11720622320902074,
            0.13009372360012994,
            0.14298122399123914,
            0.15586872438234833,
            0.16875622477345753,
            0.18164372516456662,
            0.1570874440713863,
            0.1828624448536047,
            0.2086374456358231,
            0.2344124464180415,
            0.2601874472002599,
            0.2859624479824783,
            0.31173744876469667,
            0.33751244954691506,
            0.36328745032913323,
            0.23563116610707935,
            0.27429366728040705,
            0.31295616845373475,
            0.35161866962706245,
            0.3902811708003897,
            0.4289436719737174,
            0.4676061731470451,
            0.5062686743203724,
            0.5449311754936996,
            0.42650623259564124,
            0.43939373298675044,
            0.4522812333778595,
            0.4651687337689687,
            0.4780562341600779,
            0.4909437345511871,
            0.5038312349422963,
            0.5167187353334055,
            0.5296062357245147,
            0.8530124651912825,
            0.8787874659735009,
            0.904562466755719,
            0.9303374675379374,
            0.9561124683201558,
            0.9818874691023742,
            1.0076624698845926,
            1.033437470666811,
            1.0592124714490294,
            1.2795186977869237,
            1.3181811989602514,
            1.3568437001335787,
            1.3955062013069062,
            1.4341687024802336,
            1.4728312036535613,
            1.511493704826889,
            1.5501562060002165,
            1.588818707173544,
            0.7744687431555892,
            0.7873562435466984,
            0.8002437439378076,
            0.8131312443289168,
            0.826018744720026,
            0.8389062451111352,
            0.8517937455022444,
            0.8646812458933535,
            0.8775687462844627,
            1.5489374863111784,
            1.5747124870933968,
            1.6004874878756152,
            1.6262624886578336,
            1.652037489440052,
            1.6778124902222704,
            1.7035874910044888,
            1.729362491786707,
            1.7551374925689254,
            2.3234062294667677,
            2.3620687306400954,
            2.400731231813423,
            2.4393937329867503,
            2.478056234160078,
            2.5167187353334057,
            2.555381236506733,
            2.5940437376800607,
            2.6327062388533884,
            1.1224312537155372,
            1.1353187541066465,
            1.1482062544977556,
            1.161093754888865,
            1.173981255279974,
            1.1868687556710833,
            1.1997562560621924,
            1.2126437564533017,
            1.2255312568444108,
            2.2448625074310744,
            2.270637508213293,
            2.296412508995511,
            2.32218750977773,
            2.347962510559948,
            2.3737375113421666,
            2.3995125121243848,
            2.4252875129066034,
            2.4510625136888216,
            3.3672937611466116,
            3.4059562623199393,
            3.444618763493267,
            3.4832812646665943,
            3.521943765839922,
            3.5606062670132497,
            3.599268768186577,
            3.6379312693599046,
            3.6765937705332323,
            1.4703937642754852,
            1.4832812646665945,
            1.4961687650577038,
            1.5090562654488129,
            1.521943765839922,
            1.5348312662310313,
            1.5477187666221406,
            1.5606062670132497,
            1.5734937674043588,
            2.9407875285509704,
            2.966562529333189,
            2.9923375301154076,
            3.0181125308976258,
            3.043887531679844,
            3.0696625324620626,
            3.095437533244281,
            3.1212125340264993,
            3.1469875348087175,
            4.4111812928264555,
            4.449843793999784,
            4.488506295173111,
            4.527168796346439,
            4.565831297519766,
            4.604493798693094,
            4.643156299866421,
            4.681818801039748,
            4.720481302213076,
            1.8183562748354334,
            1.8312437752265425,
            1.8441312756176518,
            1.8570187760087609,
            1.86990627639987,
            1.8827937767909793,
            1.8956812771820886,
            1.9085687775731977,
            1.9214562779643067,
            3.6367125496708668,
            3.662487550453085,
            3.6882625512353036,
            3.7140375520175217,
            3.73981255279974,
            3.7655875535819585,
            3.791362554364177,
            3.8171375551463953,
            3.8429125559286135,
            5.4550688245063,
            5.493731325679628,
            5.532393826852955,
            5.571056328026282,
            5.60971882919961,
            5.6483813303729375,
            5.687043831546266,
            5.725706332719593,
            5.76436883389292,
            2.166318785395381,
            2.1792062857864902,
            2.1920937861775998,
            2.204981286568709,
            2.2178687869598184,
            2.2307562873509275,
            2.2436437877420365,
            2.2565312881331456,
            2.2694187885242547,
            4.332637570790762,
            4.3584125715729805,
            4.3841875723551995,
            4.409962573137418,
            4.435737573919637,
            4.461512574701855,
            4.487287575484073,
            4.513062576266291,
            4.538837577048509,
            6.498956356186144,
            6.537618857359472,
            6.5762813585328,
            6.614943859706127,
            6.653606360879454,
            6.6922688620527815,
            6.73093136322611,
            6.769593864399437,
            6.808256365572765,
            2.514281295955329,
            2.5271687963464387,
            2.5400562967375477,
            2.5529437971286573,
            2.5658312975197664,
            2.5787187979108754,
            2.5916062983019845,
            2.6044937986930936,
            2.6173812990842027,
            5.028562591910658,
            5.054337592692877,
            5.0801125934750955,
            5.1058875942573145,
            5.131662595039533,
            5.157437595821751,
            5.183212596603969,
            5.208987597386187,
            5.234762598168405,
            7.542843887865988,
            7.5815063890393155,
            7.620168890212643,
            7.658831391385971,
            7.697493892559298,
            7.736156393732626,
            7.774818894905954,
            7.813481396079281,
            7.852143897252609,
        ];
        let ans = OwnedMatrixDyn::from_vec(ans, &[10, 3, 3, 3]);
        assert!(dbg!(ans - output_data).asum() < 1e-25);
        let input_grad_ans = vec![
            4.628298216795634e-19,
            4.591419346542282e-19,
            4.554540476288931e-19,
            4.51766160603558e-19,
            4.480782735782227e-19,
            4.443903865528876e-19,
            4.407024995275524e-19,
            4.370146125022172e-19,
            4.333267254768821e-19,
            9.256596433591268e-19,
            9.182838693084563e-19,
            9.109080952577862e-19,
            9.03532321207116e-19,
            8.961565471564455e-19,
            8.887807731057752e-19,
            8.814049990551047e-19,
            8.740292250044344e-19,
            8.666534509537642e-19,
            1.3884894650386903e-18,
            1.3774258039626847e-18,
            1.3663621428866795e-18,
            1.355298481810674e-18,
            1.3442348207346682e-18,
            1.3331711596586628e-18,
            1.3221074985826574e-18,
            1.311043837506652e-18,
            1.2999801764306463e-18,
            3.6325687199551393e-19,
            3.595689849701788e-19,
            3.5588109794484365e-19,
            3.5219321091950837e-19,
            3.4850532389417323e-19,
            3.4481743686883804e-19,
            3.411295498435029e-19,
            3.374416628181677e-19,
            3.3375377579283257e-19,
            7.265137439910279e-19,
            7.191379699403576e-19,
            7.117621958896873e-19,
            7.043864218390167e-19,
            6.970106477883465e-19,
            6.896348737376761e-19,
            6.822590996870058e-19,
            6.748833256363354e-19,
            6.675075515856651e-19,
            1.089770615986542e-18,
            1.0787069549105364e-18,
            1.067643293834531e-18,
            1.0565796327585252e-18,
            1.0455159716825197e-18,
            1.0344523106065141e-18,
            1.0233886495305089e-18,
            1.0123249884545033e-18,
            1.0012613273784979e-18,
            2.636839223114644e-19,
            2.5999603528612923e-19,
            2.5630814826079404e-19,
            2.526202612354589e-19,
            2.4893237421012376e-19,
            2.4524448718478857e-19,
            2.415566001594534e-19,
            2.378687131341182e-19,
            2.341808261087831e-19,
            5.273678446229288e-19,
            5.199920705722585e-19,
            5.126162965215881e-19,
            5.052405224709178e-19,
            4.978647484202475e-19,
            4.904889743695771e-19,
            4.831132003189068e-19,
            4.757374262682364e-19,
            4.683616522175662e-19,
            7.910517669343934e-19,
            7.799881058583878e-19,
            7.689244447823823e-19,
            7.578607837063768e-19,
            7.467971226303713e-19,
            7.357334615543658e-19,
            7.246698004783603e-19,
            7.136061394023547e-19,
            7.025424783263494e-19,
            1.641109726274149e-19,
            1.6042308560207974e-19,
            1.567351985767446e-19,
            1.5304731155140944e-19,
            1.4935942452607427e-19,
            1.4567153750073906e-19,
            1.4198365047540392e-19,
            1.3829576345006873e-19,
            1.3460787642473357e-19,
            3.282219452548298e-19,
            3.208461712041595e-19,
            3.134703971534892e-19,
            3.0609462310281887e-19,
            2.9871884905214854e-19,
            2.913430750014781e-19,
            2.8396730095080784e-19,
            2.7659152690013746e-19,
            2.6921575284946713e-19,
            4.923329178822447e-19,
            4.812692568062393e-19,
            4.702055957302339e-19,
            4.591419346542284e-19,
            4.480782735782228e-19,
            4.370146125022172e-19,
            4.2595095142621175e-19,
            4.1488729035020624e-19,
            4.038236292742007e-19,
            6.45380229433654e-20,
            6.085013591803025e-20,
            5.716224889269509e-20,
            5.3474361867359916e-20,
            4.978647484202475e-20,
            4.609858781668958e-20,
            4.2410700791354416e-20,
            3.872281376601925e-20,
            3.503492674068408e-20,
            1.290760458867308e-19,
            1.217002718360605e-19,
            1.1432449778539017e-19,
            1.0694872373471983e-19,
            9.95729496840495e-20,
            9.219717563337916e-20,
            8.482140158270883e-20,
            7.74456275320385e-20,
            7.006985348136816e-20,
            1.9361406883009625e-19,
            1.8255040775409078e-19,
            1.7148674667808526e-19,
            1.6042308560207976e-19,
            1.4935942452607427e-19,
            1.3829576345006875e-19,
            1.2723210237406326e-19,
            1.1616844129805777e-19,
            1.0510478022205226e-19,
            -3.503492674068408e-20,
            -3.872281376601925e-20,
            -4.2410700791354416e-20,
            -4.609858781668958e-20,
            -4.978647484202475e-20,
            -5.3474361867359916e-20,
            -5.716224889269509e-20,
            -6.085013591803025e-20,
            -6.45380229433654e-20,
            -7.006985348136816e-20,
            -7.74456275320385e-20,
            -8.482140158270883e-20,
            -9.219717563337916e-20,
            -9.95729496840495e-20,
            -1.0694872373471983e-19,
            -1.1432449778539017e-19,
            -1.217002718360605e-19,
            -1.290760458867308e-19,
            -1.0510478022205226e-19,
            -1.1616844129805777e-19,
            -1.2723210237406326e-19,
            -1.3829576345006875e-19,
            -1.4935942452607427e-19,
            -1.6042308560207976e-19,
            -1.7148674667808526e-19,
            -1.8255040775409078e-19,
            -1.9361406883009625e-19,
            -1.3460787642473357e-19,
            -1.3829576345006873e-19,
            -1.4198365047540392e-19,
            -1.4567153750073906e-19,
            -1.4935942452607427e-19,
            -1.5304731155140944e-19,
            -1.567351985767446e-19,
            -1.6042308560207974e-19,
            -1.641109726274149e-19,
            -2.6921575284946713e-19,
            -2.7659152690013746e-19,
            -2.8396730095080784e-19,
            -2.913430750014781e-19,
            -2.9871884905214854e-19,
            -3.0609462310281887e-19,
            -3.134703971534892e-19,
            -3.208461712041595e-19,
            -3.282219452548298e-19,
            -4.038236292742007e-19,
            -4.1488729035020624e-19,
            -4.2595095142621175e-19,
            -4.370146125022172e-19,
            -4.480782735782228e-19,
            -4.591419346542284e-19,
            -4.702055957302339e-19,
            -4.812692568062393e-19,
            -4.923329178822447e-19,
            -2.341808261087831e-19,
            -2.378687131341182e-19,
            -2.415566001594534e-19,
            -2.4524448718478857e-19,
            -2.4893237421012376e-19,
            -2.526202612354589e-19,
            -2.5630814826079404e-19,
            -2.5999603528612923e-19,
            -2.636839223114644e-19,
            -4.683616522175662e-19,
            -4.757374262682364e-19,
            -4.831132003189068e-19,
            -4.904889743695771e-19,
            -4.978647484202475e-19,
            -5.052405224709178e-19,
            -5.126162965215881e-19,
            -5.199920705722585e-19,
            -5.273678446229288e-19,
            -7.025424783263494e-19,
            -7.136061394023547e-19,
            -7.246698004783603e-19,
            -7.357334615543658e-19,
            -7.467971226303713e-19,
            -7.578607837063768e-19,
            -7.689244447823823e-19,
            -7.799881058583878e-19,
            -7.910517669343934e-19,
            -3.3375377579283257e-19,
            -3.374416628181677e-19,
            -3.411295498435029e-19,
            -3.4481743686883804e-19,
            -3.4850532389417323e-19,
            -3.5219321091950837e-19,
            -3.5588109794484365e-19,
            -3.595689849701788e-19,
            -3.6325687199551393e-19,
            -6.675075515856651e-19,
            -6.748833256363354e-19,
            -6.822590996870058e-19,
            -6.896348737376761e-19,
            -6.970106477883465e-19,
            -7.043864218390167e-19,
            -7.117621958896873e-19,
            -7.191379699403576e-19,
            -7.265137439910279e-19,
            -1.0012613273784979e-18,
            -1.0123249884545033e-18,
            -1.0233886495305089e-18,
            -1.0344523106065141e-18,
            -1.0455159716825197e-18,
            -1.0565796327585252e-18,
            -1.067643293834531e-18,
            -1.0787069549105364e-18,
            -1.089770615986542e-18,
            -4.333267254768821e-19,
            -4.370146125022172e-19,
            -4.407024995275524e-19,
            -4.443903865528876e-19,
            -4.480782735782227e-19,
            -4.51766160603558e-19,
            -4.554540476288931e-19,
            -4.591419346542282e-19,
            -4.628298216795634e-19,
            -8.666534509537642e-19,
            -8.740292250044344e-19,
            -8.814049990551047e-19,
            -8.887807731057752e-19,
            -8.961565471564455e-19,
            -9.03532321207116e-19,
            -9.109080952577862e-19,
            -9.182838693084563e-19,
            -9.256596433591268e-19,
            -1.2999801764306463e-18,
            -1.311043837506652e-18,
            -1.3221074985826574e-18,
            -1.3331711596586628e-18,
            -1.3442348207346682e-18,
            -1.355298481810674e-18,
            -1.3663621428866795e-18,
            -1.3774258039626847e-18,
            -1.3884894650386903e-18,
        ];
        let input_grad_ans = OwnedMatrixDyn::from_vec(input_grad_ans, &[10, 3, 3, 3]);
        let input_grad = input.get_grad().unwrap().get_data();
        assert!((input_grad_ans - input_grad).asum() < 1e-25);
        let gamma_grad_ans = vec![
            1.9984014443252818e-15,
            1.9984014443252818e-15,
            1.9984014443252818e-15,
        ];
        let gamma_grad_ans = OwnedMatrixDyn::from_vec(gamma_grad_ans, &[3]);
        let gamma_grad = gamma.get_grad().unwrap().get_data();
        assert!((gamma_grad_ans - gamma_grad).asum() < 1e-25);
        let beta_grad_ans = vec![90., 90., 90.];
        let beta_grad_ans = OwnedMatrixDyn::from_vec(beta_grad_ans, &[3]);
        let beta_grad = beta.get_grad().unwrap().get_data();
        assert!((beta_grad_ans - beta_grad).asum() < 1e-25);
    }

    #[test]
    fn batch_norm_medium_no_train_4d() {
        no_train();
        let input = (1..(10 * 3 * 3 * 3 + 1))
            .map(|x| x as f64)
            .collect::<Vec<f64>>();
        let input = OwnedMatrixDyn::from_vec(input, &[10, 3, 3, 3]);
        let mean = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let var = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let decay = OwnedMatrixDyn::from_vec(vec![0.9], &[]);
        let epsilon = OwnedMatrixDyn::from_vec(vec![1e-5], &[]);
        let gamma = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let delta = OwnedMatrixDyn::from_vec(vec![1., 2., 3.], &[3]);
        let input = crate::Variable::from(input);
        let mean = crate::Variable::from(mean);
        let var = crate::Variable::from(var);
        let decay = crate::Variable::from(decay);
        let epsilon = crate::Variable::from(epsilon);
        let gamma = crate::Variable::from(gamma);
        let beta = crate::Variable::from(delta);
        let output = batch_norm(
            mean,
            var,
            decay,
            epsilon,
            gamma.clone(),
            beta.clone(),
            input.clone(),
        );
        let output_data = output.get_data();
        let ans = vec![
            1.0,
            1.9999950000374997,
            2.9999900000749995,
            3.999985000112499,
            4.999980000149999,
            5.999975000187499,
            6.999970000224998,
            7.999965000262498,
            8.999960000299998,
            13.313680214819579,
            14.727890241672027,
            16.142100268524473,
            17.55631029537692,
            18.97052032222937,
            20.384730349081817,
            21.79894037593426,
            23.21315040278671,
            24.627360429639158,
            30.71276673319597,
            32.44481465402072,
            34.17686257484547,
            35.90891049567021,
            37.64095841649496,
            39.37300633731971,
            41.105054258144456,
            42.837102178969204,
            44.56915009979396,
            27.999865001012495,
            28.999860001049992,
            29.999855001087493,
            30.999850001124994,
            31.99984500116249,
            32.99984000119999,
            33.99983500123749,
            34.999830001274994,
            35.999825001312495,
            51.497350939835655,
            52.91156096668811,
            54.32577099354055,
            55.739981020392996,
            57.15419104724545,
            58.56840107409789,
            59.98261110095034,
            61.39682112780279,
            62.81103115465523,
            77.47806059546417,
            79.21010851628891,
            80.94215643711367,
            82.67420435793841,
            84.40625227876316,
            86.13830019958792,
            87.87034812041266,
            89.60239604123741,
            91.33444396206215,
            54.99973000202499,
            55.99972500206248,
            56.999720002099984,
            57.999715002137485,
            58.999710002174986,
            59.99970500221249,
            60.99970000224999,
            61.99969500228748,
            62.99969000232498,
            89.68102166485174,
            91.09523169170419,
            92.50944171855663,
            93.92365174540907,
            95.33786177226152,
            96.75207179911398,
            98.16628182596642,
            99.58049185281887,
            100.99470187967131,
            124.24335445773237,
            125.97540237855712,
            127.70745029938188,
            129.43949822020662,
            131.17154614103137,
            132.90359406185613,
            134.63564198268085,
            136.3676899035056,
            138.09973782433036,
            81.99959500303748,
            82.99959000307499,
            83.99958500311249,
            84.99958000314997,
            85.99957500318747,
            86.99957000322497,
            87.99956500326248,
            88.99956000329998,
            89.99955500333748,
            127.86469238986781,
            129.27890241672026,
            130.69311244357272,
            132.10732247042515,
            133.5215324972776,
            134.93574252413006,
            136.3499525509825,
            137.76416257783495,
            139.17837260468738,
            171.0086483200006,
            172.74069624082532,
            174.47274416165007,
            176.20479208247482,
            177.93684000329955,
            179.6688879241243,
            181.40093584494906,
            183.13298376577382,
            184.86503168659854,
            108.99946000404998,
            109.99945500408747,
            110.99945000412497,
            111.99944500416247,
            112.99944000419997,
            113.99943500423747,
            114.99943000427497,
            115.99942500431247,
            116.99942000434997,
            166.0483631148839,
            167.46257314173633,
            168.8767831685888,
            170.29099319544125,
            171.70520322229368,
            173.11941324914613,
            174.53362327599856,
            175.94783330285102,
            177.36204332970348,
            217.7739421822688,
            219.5059901030935,
            221.23803802391825,
            222.970085944743,
            224.70213386556776,
            226.43418178639251,
            228.16622970721727,
            229.89827762804202,
            231.63032554886672,
            135.99932500506247,
            136.99932000509997,
            137.99931500513748,
            138.99931000517498,
            139.99930500521248,
            140.99930000524998,
            141.99929500528745,
            142.99929000532495,
            143.99928500536245,
            204.23203383989997,
            205.64624386675243,
            207.06045389360486,
            208.47466392045732,
            209.88887394730975,
            211.3030839741622,
            212.71729400101466,
            214.1315040278671,
            215.54571405471955,
            264.53923604453695,
            266.2712839653617,
            268.00333188618646,
            269.7353798070112,
            271.46742772783597,
            273.1994756486607,
            274.9315235694854,
            276.6635714903102,
            278.39561941113493,
            162.99919000607497,
            163.99918500611247,
            164.99918000614997,
            165.99917500618747,
            166.99917000622497,
            167.99916500626244,
            168.99916000629995,
            169.99915500633745,
            170.99915000637495,
            242.41570456491604,
            243.8299145917685,
            245.24412461862093,
            246.6583346454734,
            248.07254467232585,
            249.48675469917828,
            250.90096472603074,
            252.31517475288317,
            253.72938477973562,
            311.30452990680516,
            313.0365778276299,
            314.76862574845467,
            316.5006736692794,
            318.2327215901042,
            319.9647695109289,
            321.69681743175363,
            323.4288653525784,
            325.16091327340314,
            189.99905500708746,
            190.99905000712496,
            191.99904500716247,
            192.99904000719994,
            193.99903500723744,
            194.99903000727494,
            195.99902500731244,
            196.99902000734994,
            197.99901500738744,
            280.59937528993214,
            282.0135853167846,
            283.427795343637,
            284.84200537048946,
            286.2562153973419,
            287.6704254241944,
            289.08463545104684,
            290.49884547789924,
            291.9130555047517,
            358.06982376907337,
            359.8018716898981,
            361.5339196107229,
            363.26596753154763,
            364.99801545237233,
            366.7300633731971,
            368.46211129402184,
            370.1941592148466,
            371.92620713567135,
            216.99892000809996,
            217.99891500813746,
            218.99891000817493,
            219.99890500821243,
            220.99890000824993,
            221.99889500828743,
            222.99889000832493,
            223.99888500836244,
            224.99888000839994,
            318.7830460149482,
            320.19725604180064,
            321.6114660686531,
            323.02567609550556,
            324.439886122358,
            325.8540961492104,
            327.2683061760629,
            328.68251620291534,
            330.0967262297678,
            404.8351176313416,
            406.56716555216633,
            408.2992134729911,
            410.03126139381584,
            411.7633093146406,
            413.49535723546524,
            415.22740515629005,
            416.95945307711474,
            418.69150099793956,
            243.99878500911245,
            244.99878000914993,
            245.99877500918743,
            246.99877000922493,
            247.99876500926243,
            248.99876000929993,
            249.99875500933743,
            250.99875000937493,
            251.99874500941243,
            356.9667167399643,
            358.38092676681674,
            359.7951367936692,
            361.2093468205216,
            362.62355684737406,
            364.0377668742265,
            365.451976901079,
            366.86618692793144,
            368.28039695478384,
            451.6004114936098,
            453.33245941443454,
            455.0645073352593,
            456.79655525608405,
            458.52860317690875,
            460.26065109773344,
            461.99269901855826,
            463.72474693938295,
            465.45679486020776,
        ];
        let ans = OwnedMatrixDyn::from_vec(ans, &[10, 3, 3, 3]);
        assert!(dbg!(ans - output_data).asum() < 1e-25);
        set_train();
    }
}

use std::collections::HashMap;

use zenu::{
    autograd::{creator::from_vec::from_vec, loss::mse::mean_squared_error, Variable},
    layer::{layers::linear::Linear, Module},
    macros::Parameters,
    matrix::{
        device::cpu::Cpu,
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
        num::Num,
    },
    optimizer::{adam::Adam, sgd::SGD, Optimizer},
};

use zenu_optimizer::adamw::AdamW;
use zenu_test::assert_val_eq;

#[derive(Parameters)]
#[parameters(num = T, device = D)]
struct SimpleNet<T, D>
where
    T: Num,
    D: Device,
{
    linear1: Linear<T, D>,
    linear2: Linear<T, D>,
}

impl<D: Device> SimpleNet<f32, D> {
    fn new() -> Self {
        use zenu::layer::Parameters;
        let (input_weights, input_bias, output_weights, output_bias) = init_parameters();
        let input_weights =
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(input_weights, DimDyn::from([4, 2]));
        let input_bias = Matrix::<Owned<f32>, DimDyn, D>::from_vec(input_bias, DimDyn::from([4]));
        let output_weights =
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(output_weights, DimDyn::from([4, 4]));
        let output_bias = Matrix::<Owned<f32>, DimDyn, D>::from_vec(output_bias, DimDyn::from([4]));

        let linear1 = Linear::new(2, 4, true);
        let linear2 = Linear::new(4, 4, true);

        let weight = &(linear1.weights().values().collect::<Vec<_>>())[0]
            .get_data_mut()
            .to_ref_mut();
        weight.copy_from(&input_weights);

        let bias = &(linear1.biases().values().collect::<Vec<_>>())[0]
            .get_data_mut()
            .to_ref_mut();
        bias.copy_from(&input_bias);

        let weight = &(linear2.weights().values().collect::<Vec<_>>())[0]
            .get_data_mut()
            .to_ref_mut();
        weight.copy_from(&output_weights);

        let bias = &(linear2.biases().values().collect::<Vec<_>>())[0]
            .get_data_mut()
            .to_ref_mut();
        bias.copy_from(&output_bias);

        Self { linear1, linear2 }
    }
}

impl<T: Num, D: Device> Module<T, D> for SimpleNet<T, D> {
    type Input = Variable<T, D>;
    type Output = Variable<T, D>;

    fn call(&self, input: Self::Input) -> Self::Output {
        let x = self.linear1.call(input);
        self.linear2.call(x)
    }
}

fn init_parameters() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let input_parameters = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.07, 0.08];
    let input_bias = vec![0.1, 0.2, 0.3, 0.4];

    let output_parameters = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
    ];
    let output_bias = vec![0.1, 0.2, 0.3, 0.4];

    (input_parameters, input_bias, output_parameters, output_bias)
}

fn test_funcion_inner<O: Optimizer<f32, Cpu, SimpleNet<f32, Cpu>>>(
    net: &SimpleNet<f32, Cpu>,
    optimizer: &O,
) -> HashMap<String, Variable<f32, Cpu>> {
    use zenu::layer::Parameters;

    let input = from_vec(vec![0.1, 0.2], DimDyn::from([1, 2]));
    let target = from_vec(vec![0.1, 0.2, 0.3, 0.4], DimDyn::from([4]));
    let output = net.call(input);
    let loss = mean_squared_error(target, output);
    loss.backward();
    optimizer.update(net);
    loss.clear_grad();
    net.parameters()
}

#[test]
fn sgd_test() {
    let net = SimpleNet::<f32, Cpu>::new();
    let optimizer = SGD::new(0.9);
    let parameters = test_funcion_inner(&net, &optimizer);

    let ans_linear1 = vec![
        0.0891, 0.1782, 0.2857, 0.3715, 0.4917, 0.5833, 0.0596, 0.0592,
    ];
    let ans_linear1 =
        Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(ans_linear1, DimDyn::from([4, 2]));
    let ans_bias1 = vec![-0.0089, 0.0574, 0.2166, 0.2961];
    let ans_bias1 = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(ans_bias1, DimDyn::from([4]));
    let ans_linear2 = vec![
        0.0739, 0.1460, 0.2181, 0.3263, 0.4779, 0.5543, 0.0007, 0.0176, 0.0801, 0.0795, 0.0789,
        0.0920, 0.1164, 0.1119, 0.1075, 0.1217,
    ];
    let ans_linear2 =
        Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(ans_linear2, DimDyn::from([4, 4]));
    let ans_bias2 = vec![-0.0742, 0.0525, 0.2339, 0.3095];
    let ans_bias2 = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(ans_bias2, DimDyn::from([4]));

    assert_val_eq!(
        parameters["linear1.linear.weight"].clone(),
        ans_linear1,
        1e-4
    );
    assert_val_eq!(parameters["linear1.linear.bias"].clone(), ans_bias1, 1e-4);
    assert_val_eq!(
        parameters["linear2.linear.weight"].clone(),
        ans_linear2,
        1e-4
    );
    assert_val_eq!(parameters["linear2.linear.bias"].clone(), ans_bias2, 1e-4);
}

#[test]
fn adam_test() {
    use zenu::layer::Parameters;
    let net = SimpleNet::<f32, Cpu>::new();
    let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8, &net);
    let _ = test_funcion_inner(&net, &optimizer);
    let mut net_ = SimpleNet::<f32, Cpu>::new();
    net_.load_parameters(net.parameters());
    let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8, &net_);
    let parameters = test_funcion_inner(&net_, &optimizer);
    let linear1_weight = vec![
        0.0801, 0.1801, 0.2801, 0.3801, 0.4801, 0.5801, 0.0501, 0.0601,
    ];
    let linear1_weight =
        Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(linear1_weight, DimDyn::from([4, 2]));
    let linear1_bias = vec![0.0801, 0.1801, 0.2801, 0.3801];
    let linear1_bias = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(linear1_bias, DimDyn::from([4]));
    let linear2_weight = vec![
        0.0801, 0.1801, 0.2801, 0.3801, 0.4801, 0.5801, 0.0501, 0.0601, 0.0702, 0.0801, 0.0901,
        0.1001, 0.1101, 0.1201, 0.1301, 0.1401,
    ];
    let linear2_weight =
        Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(linear2_weight, DimDyn::from([4, 4]));
    let linear2_bias = vec![0.0800, 0.1801, 0.2801, 0.3801];
    let linear2_bias = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(linear2_bias, DimDyn::from([4]));

    assert_val_eq!(
        parameters["linear1.linear.weight"].clone(),
        linear1_weight,
        2e-4
    );
    assert_val_eq!(
        parameters["linear1.linear.bias"].clone(),
        linear1_bias,
        2e-4
    );
    assert_val_eq!(
        parameters["linear2.linear.weight"].clone(),
        linear2_weight,
        2e-4
    );
    assert_val_eq!(
        parameters["linear2.linear.bias"].clone(),
        linear2_bias,
        2e-4
    );
}

#[test]
fn adam_w_test() {
    let net = SimpleNet::<f32, Cpu>::new();
    let optimizer = AdamW::new(0.01, 0.9, 0.999, 1e-8, 0.01, &net);
    let _ = test_funcion_inner(&net, &optimizer);
    let parameters = test_funcion_inner(&net, &optimizer);
    let linear1_weight = vec![
        0.0801, 0.1800, 0.2800, 0.3800, 0.4800, 0.5800, 0.0501, 0.0601,
    ];
    let linear1_weight =
        Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(linear1_weight, DimDyn::from([4, 2]));
    let linear1_bias = vec![0.0801, 0.1800, 0.2800, 0.3800];
    let linear1_bias = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(linear1_bias, DimDyn::from([4]));
    let linear2_weight = vec![
        0.0801, 0.1800, 0.2800, 0.3800, 0.4800, 0.5800, 0.0501, 0.0601, 0.0702, 0.0801, 0.0901,
        0.1001, 0.1101, 0.1201, 0.1301, 0.1401,
    ];
    let linear2_weight =
        Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(linear2_weight, DimDyn::from([4, 4]));
    let linear2_bias = vec![0.0800, 0.1800, 0.2801, 0.3800];
    let linear2_bias = Matrix::<Owned<f32>, DimDyn, Cpu>::from_vec(linear2_bias, DimDyn::from([4]));
    assert_val_eq!(
        parameters["linear1.linear.weight"].clone(),
        linear1_weight,
        2e-4
    );
    assert_val_eq!(
        parameters["linear1.linear.bias"].clone(),
        linear1_bias,
        2e-4
    );
    assert_val_eq!(
        parameters["linear2.linear.weight"].clone(),
        linear2_weight,
        2e-4
    );
    assert_val_eq!(
        parameters["linear2.linear.bias"].clone(),
        linear2_bias,
        2e-4
    );
}

use zenu::layer::layers::linear::Linear;
use zenu::macros::Parameters as ParametersDerive;
use zenu::matrix::{
    device::{cpu::Cpu, Device},
    matrix::Matrix,
    num::Num,
};
use zenu_test::assert_val_eq;

#[derive(ParametersDerive)]
#[parameters(num = T, device = D)]
pub struct Hoge<T, D>
where
    T: Num,
    D: Device,
{
    pub linear: Linear<T, D>,
}

#[test]
fn small_net() {
    use zenu::layer::Parameters;
    let hoge = Hoge::<f32, Cpu> {
        linear: Linear::new(2, 2, true),
    };

    let weights = hoge.weights();
    let biases = hoge.biases();
    let parameters = hoge.parameters();

    assert_eq!(weights.len(), 1);
    assert_eq!(biases.len(), 1);
    assert_eq!(parameters.len(), 2);

    assert_val_eq!(
        weights["linear.linear.weight"].clone(),
        hoge.linear.weight.get_data(),
        1e-4
    );

    let linear_bias = hoge.linear.bias.clone().unwrap();
    let linear_bias = linear_bias.get_data();

    assert_val_eq!(
        biases["linear.linear.bias"].clone(),
        linear_bias.clone(),
        1e-4
    );

    assert_val_eq!(
        parameters["linear.linear.weight"].clone(),
        hoge.linear.weight.get_data(),
        1e-4
    );
    assert_val_eq!(parameters["linear.linear.bias"].clone(), linear_bias, 1e-4);
}

#[test]
fn test_load_parameters() {
    use zenu::layer::Parameters;
    let base_model = Hoge::<f32, Cpu> {
        linear: Linear::new(2, 2, true),
    };

    let base_model_parameters = base_model.parameters();

    let mut new_model = Hoge::<f32, Cpu> {
        linear: Linear::new(2, 2, true),
    };

    let new_model_weight = new_model.linear.weight.get_as_ref();

    println!("{:?}", base_model_parameters.keys());

    println!(
        "new_model.parameters().keys(): {:?}",
        new_model.parameters().keys()
    );

    new_model
        .linear
        .weight
        .get_as_mut()
        .copy_from(&Matrix::zeros_like(&new_model_weight));

    let new_model_bias = new_model.linear.bias.clone().unwrap().get_as_ref();
    new_model
        .linear
        .bias
        .clone()
        .unwrap()
        .get_as_mut()
        .copy_from(&Matrix::zeros_like(&new_model_bias));

    new_model.load_parameters(base_model_parameters);
}

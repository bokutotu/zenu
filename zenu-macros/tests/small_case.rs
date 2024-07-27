use zenu_layer::{layers::linear::Linear, Parameters};
use zenu_macros::Parameters;
use zenu_matrix::{device::cpu::Cpu, device::Device, num::Num};
use zenu_test::assert_val_eq;

#[derive(Parameters)]
pub struct Hoge<T, D>
where
    T: Num,
    D: Device,
{
    pub linear: Linear<T, D>,
}

#[test]
fn small_net() {
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

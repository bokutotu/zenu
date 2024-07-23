use serde::{Deserialize, Serialize};
use zenu_layer::{layers::linear::Linear, Parameteres};
use zenu_macros::ZenuModel;
use zenu_matrix::{device::cpu::Cpu, device::Device, num::Num};

#[derive(ZenuModel, Serialize, Deserialize)]
#[serde(bound(deserialize = "T: Num + Deserialize<'de>"))]
pub struct Hoge<T: Num, D: Device> {
    pub linear: Linear<T, D>,
}

#[test]
fn test_zenu_model() {
    let hoge = Hoge::<f32, Cpu> {
        linear: Linear::new(10, 10, true),
    };

    let weights = hoge.weights();
    let biases = hoge.biases();

    let ans_weights = hoge.linear.weight.clone();
    let ans_weights = ans_weights.get_data();
    let ans_biases = hoge.linear.bias.clone().unwrap();
    let ans_biases = ans_biases.get_data();

    let weight = weights[0].get_data();
    let bias = biases[0].get_data();

    // Add assertions here to test the behavior
    assert_eq!((weight.to_ref() - ans_weights.to_ref()).asum(), 0.);
    assert_eq!((bias.to_ref() - ans_biases.to_ref()).asum(), 0.);
}

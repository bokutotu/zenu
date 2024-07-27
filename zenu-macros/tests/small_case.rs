use zenu_layer::{layers::linear::Linear, Parameters};
use zenu_macros::Parameters;
use zenu_matrix::{device::cpu::Cpu, device::Device, num::Num};

#[derive(Parameters)]
pub struct Hoge<T, D>
where
    T: Num,
    D: Device,
{
    pub linear: Linear<T, D>,
}

use crate::{
    device::{cpu::Cpu, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref, Repr},
    num::Num,
};

use rand::seq::SliceRandom;
#[cfg(feature = "nvidia")]
use zenu_cuda::cudnn::dropout::DropoutConfig;

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

#[cfg(feature = "nvidia")]
use super::NNCache;

pub struct DropoutState<T: Num, D: DeviceBase> {
    pub rate: f32,
    pub state: Option<Matrix<Owned<T>, DimDyn, Cpu>>,
    #[cfg(feature = "nvidia")]
    gpu_state: Option<DropoutConfig<T>>,
    #[cfg(feature = "nvidia")]
    state_cache: Option<NNCache<D>>,
    #[cfg(feature = "nvidia")]
    space_cache: Option<NNCache<D>>,
    _device: std::marker::PhantomData<D>,
}

impl<T: Num, D: DeviceBase> DropoutState<T, D> {
    pub fn new(rate: f32) -> Self {
        Self {
            rate,
            state: None,
            #[cfg(feature = "nvidia")]
            gpu_state: None,
            #[cfg(feature = "nvidia")]
            state_cache: None,
            #[cfg(feature = "nvidia")]
            space_cache: None,
            _device: std::marker::PhantomData,
        }
    }

    #[allow(unused_variables)]
    pub fn gpu_init(&mut self, shape: DimDyn) {
        #[cfg(feature = "nvidia")]
        {
            let gpu_state = DropoutConfig::new(shape.slice()).unwrap();
            let state_size = gpu_state.get_state_size();
            let cache = NNCache::<D>::new(state_size);
            gpu_state.set(self.rate, 0, cache.ptr as *mut _).unwrap();
            self.gpu_state = Some(gpu_state);
            self.state_cache = Some(cache);
        }
        #[cfg(not(feature = "nvidia"))]
        {
            panic!("GPU support is not enabled");
        }
    }
}

fn dropout_mask_inner<T>(input: &mut [T], dropout_ratio: f32)
where
    T: Copy + std::ops::Mul<T, Output = T> + std::ops::AddAssign + Num,
{
    assert!(
        (0.0..=1.0).contains(&dropout_ratio),
        "Dropout ratio must be between 0 and 1"
    );

    let num_zeros_item = (input.len() as f32 * dropout_ratio) as usize;
    let mut rng = rand::thread_rng();

    let mut indices: Vec<usize> = (0..input.len()).collect();
    indices.shuffle(&mut rng);
    for i in 0..num_zeros_item {
        input[indices[i]] = T::zero();
    }
}

fn dropout_mask<T>(input: Matrix<Ref<&mut T>, DimDyn, Cpu>, dropout_ratio: f32)
where
    T: Copy + std::ops::Mul<T, Output = T> + std::ops::AddAssign + Num,
{
    let input_slice = input.as_mut_slice();
    dropout_mask_inner(input_slice, dropout_ratio);
}

pub trait Dropout: DeviceBase {
    fn dropout<T: Num>(
        x: &Matrix<Ref<&T>, DimDyn, Self>,
        state: &mut DropoutState<T, Self>,
    ) -> Matrix<Owned<T>, DimDyn, Self>;

    fn dropout_grad<T: Num>(
        dy: &Matrix<Ref<&T>, DimDyn, Self>,
        state: &DropoutState<T, Self>,
    ) -> Matrix<Owned<T>, DimDyn, Self>;
}

impl Dropout for Cpu {
    fn dropout<T: Num>(
        x: &Matrix<Ref<&T>, DimDyn, Self>,
        state: &mut DropoutState<T, Self>,
    ) -> Matrix<Owned<T>, DimDyn, Self> {
        let rate = state.rate;
        let num_elm = x.shape().num_elm();
        let mask = {
            let mut mask = Matrix::alloc(&[num_elm]);
            dropout_mask(mask.to_ref_mut(), rate);
            mask
        };
        let y = x.reshape([num_elm]) * mask.to_ref();
        state.state = Some(mask);
        y.reshape_no_alloc_owned(x.shape())
    }

    fn dropout_grad<T: Num>(
        dy: &Matrix<Ref<&T>, DimDyn, Self>,
        state: &DropoutState<T, Self>,
    ) -> Matrix<Owned<T>, DimDyn, Self> {
        let rate = state.rate;
        let mask = state.state.as_ref().unwrap();
        let dx = (dy.to_ref() * mask.to_ref()) * (T::one() / T::from(1.0 - rate).unwrap());
        dx.reshape_no_alloc_owned(dy.shape())
    }
}

#[cfg(feature = "nvidia")]
impl Dropout for Nvidia {
    fn dropout<T: Num>(
        x: &Matrix<Ref<&T>, DimDyn, Self>,
        state: &mut DropoutState<T, Self>,
    ) -> Matrix<Owned<T>, DimDyn, Self> {
        match state.gpu_state {
            Some(_) => {}
            None => {
                state.gpu_init(x.shape());
            }
        };

        match state.space_cache {
            Some(_) => {}
            None => {
                let space_cache = NNCache::<Self>::new(0);
                state.space_cache = Some(space_cache);
            }
        };

        let mut y = Matrix::alloc(x.shape().slice());
        let space_cache = state.space_cache.as_ref().unwrap();
        let spcace_cache_ptr = space_cache.ptr as *mut _;

        {
            let y_mut_ref = y.to_ref_mut();
            state
                .gpu_state
                .as_ref()
                .unwrap()
                .forward(
                    x.as_ptr() as *const _,
                    y_mut_ref.as_mut_ptr() as *mut _,
                    spcace_cache_ptr,
                )
                .unwrap();
        }
        y
    }

    fn dropout_grad<T: Num>(
        dy: &Matrix<Ref<&T>, DimDyn, Self>,
        state: &DropoutState<T, Self>,
    ) -> Matrix<Owned<T>, DimDyn, Self> {
        let gpu_state = state.gpu_state.as_ref().unwrap();
        let space_cache = state.space_cache.as_ref().unwrap();

        let mut dx = Matrix::alloc(dy.shape().slice());

        {
            let dx_mut_ref = dx.to_ref_mut();
            gpu_state
                .backward(
                    dy.as_ptr() as *const _,
                    dx_mut_ref.as_mut_ptr() as *mut _,
                    space_cache.ptr as *mut _,
                )
                .unwrap();
        }
        dx
    }
}

pub fn dropout<R, D>(
    x: &Matrix<R, DimDyn, D>,
    state: &mut DropoutState<R::Item, D>,
) -> Matrix<Owned<R::Item>, DimDyn, D>
where
    R: Repr,
    D: Dropout + DeviceBase,
{
    if x.shape().len() != 2 && x.shape().len() != 4 {
        panic!("Only 2D and 4D tensors are supported");
    }
    D::dropout(&x.to_ref(), state)
}

pub fn dropout_grad<R, D>(
    dy: &Matrix<R, DimDyn, D>,
    state: &DropoutState<R::Item, D>,
) -> Matrix<Owned<R::Item>, DimDyn, D>
where
    R: Repr,
    D: Dropout + DeviceBase,
{
    if dy.shape().len() != 2 && dy.shape().len() != 4 {
        panic!("Only 2D and 4D tensors are supported");
    }
    D::dropout_grad(&dy.to_ref(), state)
}

#[cfg(test)]
mod dropout {
    use zenu_test::run_mat_test;

    use crate::{
        device::{cpu::Cpu, Device},
        matrix::Matrix,
    };

    use super::{dropout_grad, DropoutState};

    fn dropout_4d<D: Device>() {
        let mut state = DropoutState::<f32, D>::new(0.8);
        let x = crate::matrix::Matrix::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[1, 2, 2, 3],
        );

        let y = super::dropout(&x, &mut state);
        let y_cpu = y.clone().to::<Cpu>();
        let y_cpu_ref = y_cpu.to_ref();
        let y_cpu_slice = y_cpu_ref.as_slice();
        let zero_indexed = y_cpu_slice.iter().map(|x| *x == 0.).collect::<Vec<bool>>();
        let y_grad = Matrix::ones_like(&y);
        let x_grad = dropout_grad(&y_grad.to_ref(), &state);
        let x_grad_cpu = x_grad.to::<Cpu>();
        let x_grad_cpu_ref = x_grad_cpu.to_ref();
        let x_grad_cpu_slice = x_grad_cpu_ref.as_slice();

        for i in 0..y_cpu_slice.len() {
            if zero_indexed[i] {
                assert_eq!(x_grad_cpu_slice[i], 0.0);
            } else {
                assert_eq!(x_grad_cpu_slice[i], 1. / (1. - 0.8));
            }
        }
    }
    run_mat_test!(dropout_4d, dropout_4d_cpu, dropout_4d_gpu);
}

use std::any::TypeId;

use crate::{
    device::{cpu::Cpu, Device, DeviceBase},
    dim::{DimDyn, DimTrait},
    index::index_dyn_impl::Index,
    matrix::{Matrix, Owned, Repr},
    num::Num,
};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

#[cfg(feature = "nvidia")]
use zenu_cuda::cublas::*;

pub trait MaxIdx: DeviceBase {
    fn max_idx<T: Num>(input: *const T, size: usize, stride: usize) -> usize;
}

impl MaxIdx for Cpu {
    fn max_idx<T: Num>(input: *const T, size: usize, stride: usize) -> usize {
        extern crate openblas_src;
        use cblas::*;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let input = input as *const f32;
            let input = unsafe { std::slice::from_raw_parts(input, size * stride) };
            unsafe { isamax(size as i32, input, stride as i32) as usize }
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let input = input as *const f64;
            let input = unsafe { std::slice::from_raw_parts(input, size * stride) };
            unsafe { idamax(size as i32, input, stride as i32) as usize }
        } else {
            panic!("Unsupported type");
        }
    }
}

#[cfg(feature = "nvidia")]
impl MaxIdx for Nvidia {
    fn max_idx<T: Num>(input: *const T, size: usize, stride: usize) -> usize {
        cublas_amax(size, input, stride)
            .unwrap()
            .try_into()
            .unwrap()
    }
}

impl<T: Num, R: Repr<Item = T>, D: Device> Matrix<R, DimDyn, D> {
    pub fn max_idx(&self) -> DimDyn {
        let default_stride = self.to_default_stride();
        let idx = <D as MaxIdx>::max_idx(
            default_stride.as_ptr(),
            default_stride.shape().num_elm(),
            default_stride.stride()[default_stride.shape().len() - 1],
        );
        default_stride.shape_stride().get_dim_by_offset(idx)
    }

    pub fn max_item(&self) -> T {
        let idx = self.max_idx();
        self.index_item(idx)
    }

    /// selfはdefault stride
    pub fn max_axis(&self, axis: usize) -> Matrix<Owned<T>, DimDyn, D> {
        if axis >= self.shape().len() {
            panic!("max_axis: Axis out of bounds");
        }

        let mut output_shape = Vec::new();
        for i in 0..self.shape().len() {
            if i == axis {
                continue;
            }
            output_shape.push(self.shape()[i]);
        }

        let output_shape = DimDyn::from(&output_shape as &[usize]);
        let mut output = Matrix::<Owned<T>, DimDyn, D>::zeros(output_shape);

        if axis == 0 {
            let output_flatten = output.reshape_mut([output.shape().num_elm()]);
            let s = self.reshape_new_matrix([self.shape()[0], output_shape.num_elm()]);
            for i in 0..output_shape.num_elm() {
                output_flatten.index_item_assign(&[i], s.index_axis(Index::new(1, i)).max_item());
            }
        } else {
            for i in 0..self.shape()[0] {
                let s = self.index_axis(Index::new(0, i));
                let output = output.to_ref_mut().index_axis_mut(Index::new(0, i));
                output.copy_from(&s.max_axis(axis - 1));
            }
        }

        output
    }
}

#[cfg(test)]
mod max_idx {
    use crate::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
        slice_dynamic,
    };

    use zenu_test::{assert_mat_eq_epsilon, run_mat_test};

    fn default_1d<D: Device>() {
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![0., 1., 2., 3.], [4]);
        assert_eq!(a.to_ref().max_idx(), [3].into());
    }
    #[test]
    fn default_1d_cpu() {
        default_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn default_1d_gpu() {
        default_1d::<crate::device::nvidia::Nvidia>();
    }

    fn default_2d<D: Device>() {
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![0., 1., 2., 3.], [2, 2]);
        assert_eq!(a.to_ref().max_idx(), [1, 1].into());
    }
    #[test]
    fn default_2d_cpu() {
        default_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn default_2d_gpu() {
        default_2d::<crate::device::nvidia::Nvidia>();
    }

    fn sliced_3d<D: Device>() {
        let mut v = Vec::new();
        for i in 0..8 * 8 * 8 {
            v.push(i as f32);
        }
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(v, [8, 8, 8]);
        let sliced = a.slice(slice_dynamic!(..;3, ..;4, ..;2));
        assert_eq!(sliced.max_idx(), [2, 1, 3].into());
    }
    #[test]
    fn sliced_3d_cpu() {
        sliced_3d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sliced_3d_gpu() {
        sliced_3d::<crate::device::nvidia::Nvidia>();
    }

    fn max_axis_4d<D: Device>() {
        let input: Vec<f32> = vec![
            0.5488135039273248,
            0.7151893663724195,
            0.6027633760716439,
            0.5448831829968969,
            0.4236547993389047,
            0.6458941130666561,
            0.4375872112626925,
            0.8917730007820798,
            0.9636627605010293,
            0.3834415188257777,
            0.7917250380826646,
            0.5288949197529045,
            0.5680445610939323,
            0.925596638292661,
            0.07103605819788694,
            0.08712929970154071,
            0.02021839744032572,
            0.832619845547938,
            0.7781567509498505,
            0.8700121482468192,
            0.978618342232764,
            0.7991585642167236,
            0.46147936225293185,
            0.7805291762864555,
            0.11827442586893322,
            0.6399210213275238,
            0.1433532874090464,
            0.9446689170495839,
            0.5218483217500717,
            0.4146619399905236,
            0.26455561210462697,
            0.7742336894342167,
            0.45615033221654855,
            0.5684339488686485,
            0.018789800436355142,
            0.6176354970758771,
            0.6120957227224214,
            0.6169339968747569,
            0.9437480785146242,
            0.6818202991034834,
            0.359507900573786,
            0.43703195379934145,
            0.6976311959272649,
            0.06022547162926983,
            0.6667667154456677,
            0.6706378696181594,
            0.2103825610738409,
            0.1289262976548533,
            0.31542835092418386,
            0.3637107709426226,
            0.5701967704178796,
            0.43860151346232035,
            0.9883738380592262,
            0.10204481074802807,
            0.2088767560948347,
            0.16130951788499626,
            0.6531083254653984,
            0.2532916025397821,
            0.4663107728563063,
            0.24442559200160274,
            0.15896958364551972,
            0.11037514116430513,
            0.6563295894652734,
            0.1381829513486138,
            0.1965823616800535,
            0.3687251706609641,
            0.8209932298479351,
            0.09710127579306127,
            0.8379449074988039,
            0.09609840789396307,
            0.9764594650133958,
            0.4686512016477016,
            0.9767610881903371,
            0.604845519745046,
            0.7392635793983017,
            0.039187792254320675,
            0.2828069625764096,
            0.1201965612131689,
            0.29614019752214493,
            0.11872771895424405,
            0.317983179393976,
            0.41426299451466997,
            0.06414749634878436,
            0.6924721193700198,
            0.5666014542065752,
            0.2653894909394454,
            0.5232480534666997,
            0.09394051075844168,
            0.5759464955561793,
            0.9292961975762141,
            0.31856895245132366,
            0.6674103799636817,
            0.13179786240439217,
            0.7163272041185655,
            0.2894060929472011,
            0.18319136200711683,
            0.5865129348100832,
            0.020107546187493552,
            0.8289400292173631,
            0.004695476192547066,
            0.6778165367962301,
            0.27000797319216485,
            0.7351940221225949,
            0.9621885451174382,
            0.24875314351995803,
            0.5761573344178369,
            0.592041931271839,
            0.5722519057908734,
            0.2230816326406183,
            0.952749011516985,
            0.44712537861762736,
            0.8464086724711278,
            0.6994792753175043,
            0.29743695085513366,
            0.8137978197024772,
            0.39650574084698464,
            0.8811031971111616,
            0.5812728726358587,
            0.8817353618548528,
            0.6925315900777659,
        ];
        let ans_0d = vec![
            0.5488135039273248,
            0.7151893663724195,
            0.6563295894652734,
            0.5448831829968969,
            0.4236547993389047,
            0.6458941130666561,
            0.8209932298479351,
            0.8917730007820798,
            0.9636627605010293,
            0.3834415188257777,
            0.9764594650133958,
            0.5288949197529045,
            0.9767610881903371,
            0.925596638292661,
            0.7392635793983017,
            0.08712929970154071,
            0.2828069625764096,
            0.832619845547938,
            0.7781567509498505,
            0.8700121482468192,
            0.978618342232764,
            0.7991585642167236,
            0.46147936225293185,
            0.7805291762864555,
            0.5666014542065752,
            0.6399210213275238,
            0.5232480534666997,
            0.9446689170495839,
            0.5759464955561793,
            0.9292961975762141,
            0.31856895245132366,
            0.7742336894342167,
            0.45615033221654855,
            0.7163272041185655,
            0.2894060929472011,
            0.6176354970758771,
            0.6120957227224214,
            0.6169339968747569,
            0.9437480785146242,
            0.6818202991034834,
            0.6778165367962301,
            0.43703195379934145,
            0.7351940221225949,
            0.9621885451174382,
            0.6667667154456677,
            0.6706378696181594,
            0.592041931271839,
            0.5722519057908734,
            0.31542835092418386,
            0.952749011516985,
            0.5701967704178796,
            0.8464086724711278,
            0.9883738380592262,
            0.29743695085513366,
            0.8137978197024772,
            0.39650574084698464,
            0.8811031971111616,
            0.5812728726358587,
            0.8817353618548528,
            0.6925315900777659,
        ];
        let ans_1d = vec![
            0.978618342232764,
            0.7991585642167236,
            0.6976311959272649,
            0.7805291762864555,
            0.6667667154456677,
            0.6706378696181594,
            0.4375872112626925,
            0.9446689170495839,
            0.9636627605010293,
            0.4146619399905236,
            0.7917250380826646,
            0.7742336894342167,
            0.9883738380592262,
            0.925596638292661,
            0.2088767560948347,
            0.6176354970758771,
            0.6531083254653984,
            0.832619845547938,
            0.9437480785146242,
            0.8700121482468192,
            0.6778165367962301,
            0.41426299451466997,
            0.7351940221225949,
            0.9621885451174382,
            0.5666014542065752,
            0.5761573344178369,
            0.8209932298479351,
            0.5722519057908734,
            0.8379449074988039,
            0.952749011516985,
            0.9764594650133958,
            0.8464086724711278,
            0.9767610881903371,
            0.7163272041185655,
            0.8137978197024772,
            0.39650574084698464,
            0.8811031971111616,
            0.5812728726358587,
            0.8817353618548528,
            0.6925315900777659,
        ];
        let ans_2d = vec![
            0.7917250380826646,
            0.7151893663724195,
            0.8917730007820798,
            0.9636627605010293,
            0.8700121482468192,
            0.978618342232764,
            0.7991585642167236,
            0.9446689170495839,
            0.9437480785146242,
            0.6818202991034834,
            0.6706378696181594,
            0.6531083254653984,
            0.9883738380592262,
            0.4663107728563063,
            0.6667667154456677,
            0.9764594650133958,
            0.8209932298479351,
            0.9767610881903371,
            0.8379449074988039,
            0.7392635793983017,
            0.31856895245132366,
            0.6674103799636817,
            0.13179786240439217,
            0.8289400292173631,
            0.9292961975762141,
            0.6778165367962301,
            0.8811031971111616,
            0.7351940221225949,
            0.9621885451174382,
            0.952749011516985,
        ];

        let ans_3d = vec![
            0.7151893663724195,
            0.9636627605010293,
            0.925596638292661,
            0.8700121482468192,
            0.978618342232764,
            0.9446689170495839,
            0.7742336894342167,
            0.9437480785146242,
            0.6976311959272649,
            0.6706378696181594,
            0.9883738380592262,
            0.6531083254653984,
            0.6563295894652734,
            0.8379449074988039,
            0.9767610881903371,
            0.29614019752214493,
            0.6924721193700198,
            0.9292961975762141,
            0.7163272041185655,
            0.8289400292173631,
            0.9621885451174382,
            0.952749011516985,
            0.8464086724711278,
            0.8817353618548528,
        ];

        let input = Matrix::<Owned<f32>, DimDyn, D>::from_vec(input, [2, 3, 4, 5]);
        let ans_0d = Matrix::<Owned<f32>, DimDyn, D>::from_vec(ans_0d, [3, 4, 5]);
        let ans_1d = Matrix::<Owned<f32>, DimDyn, D>::from_vec(ans_1d, [2, 4, 5]);
        let ans_2d = Matrix::<Owned<f32>, DimDyn, D>::from_vec(ans_2d, [2, 3, 5]);
        let ans_3d = Matrix::<Owned<f32>, DimDyn, D>::from_vec(ans_3d, [2, 3, 4]);

        assert_mat_eq_epsilon!(input.max_axis(0), ans_0d, 1e-6);
        assert_mat_eq_epsilon!(input.max_axis(1), ans_1d, 1e-6);
        assert_mat_eq_epsilon!(input.max_axis(2), ans_2d, 1e-6);
        assert_mat_eq_epsilon!(input.max_axis(3), ans_3d, 1e-6);
    }
    run_mat_test!(max_axis_4d, max_axis_4d_cpu, max_axis_4d_gpu);
}

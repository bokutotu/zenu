use std::ptr::NonNull;

use zenu_cudnn_frontend_wrapper_sys::{
    check_conv_graph, ConvBkwdDataBuffers, ConvBkwdDataDescriptor, ConvBkwdFilterBuffers,
    ConvBkwdFilterDescriptor, ConvBufers, ConvDescriptor, ConvInfo,
};

use zenu_cudnn_sys::cudnnHandle_t;

use super::graph_utils::{get_cudnn_frontend_type, shape_stride_to_cudnn, success_or_panic};

#[expect(clippy::cast_possible_wrap)]
fn vec_to_slice_2(v: &[usize]) -> [i64; 2] {
    assert!(v.len() <= 2, "Vec length is greater than 2");

    let mut arr = [0; 2];
    for i in 0..v.len() {
        arr[i] = v[i] as i64;
    }
    arr
}

#[expect(clippy::cast_possible_wrap)]
fn get_conv_info(pad: &[usize], stride: &[usize], dilation: &[usize]) -> ConvInfo {
    let num_dims = pad.len() as i64;
    let padding = vec_to_slice_2(pad);
    let stride = vec_to_slice_2(stride);
    let dilation = vec_to_slice_2(dilation);

    ConvInfo {
        padding,
        stride,
        dilation,
        num_dims,
    }
}

pub struct ConvForwardGraph(NonNull<ConvDescriptor>);

impl ConvForwardGraph {
    #[expect(clippy::too_many_arguments)]
    fn new<T>(
        x_shape: &[usize],
        x_stride: &[usize],
        w_shape: &[usize],
        w_stride: &[usize],
        y_shape: &[usize],
        y_stride: &[usize],
        pad: &[usize],
        stride: &[usize],
        dilation: &[usize],
    ) -> Self {
        let mut x_shape_stride = shape_stride_to_cudnn(x_shape, x_stride);
        let mut w_shape_stride = shape_stride_to_cudnn(w_shape, w_stride);
        let mut y_shape_stride = shape_stride_to_cudnn(y_shape, y_stride);
        let mut conv_info = get_conv_info(pad, stride, dilation);
        let data_type = get_cudnn_frontend_type::<T>();

        let mut inner = std::ptr::null_mut();
        let status = unsafe {
            zenu_cudnn_frontend_wrapper_sys::create_conv_descriptor(
                std::ptr::from_mut(&mut inner),
                data_type,
                std::ptr::from_mut(&mut x_shape_stride),
                std::ptr::from_mut(&mut w_shape_stride),
                std::ptr::from_mut(&mut y_shape_stride),
                std::ptr::from_mut(&mut conv_info),
            )
        };
        success_or_panic(status);
        Self(NonNull::new(inner).unwrap())
    }

    fn check_graph(&self, handle: *mut cudnnHandle_t) {
        let status = unsafe { check_conv_graph(self.0.as_ptr(), handle.cast()) };
        success_or_panic(status);
    }

    #[must_use]
    pub fn get_workspace_size(&self) -> usize {
        let mut size = 0;
        let status = unsafe {
            zenu_cudnn_frontend_wrapper_sys::get_conv_workspace_size(
                self.0.as_ptr(),
                std::ptr::from_mut(&mut size),
            )
        };
        success_or_panic(status);
        size.try_into().unwrap()
    }

    pub fn execute<T>(
        &self,
        x: *mut T,
        w: *mut T,
        y: *mut T,
        workspace: *mut T,
        handle: *mut cudnnHandle_t,
    ) {
        let mut buf = ConvBufers {
            X: x.cast(),
            filter: w.cast(),
            Y: y.cast(),
        };
        let status = unsafe {
            zenu_cudnn_frontend_wrapper_sys::execute_conv_forward(
                self.0.as_ptr(),
                std::ptr::from_mut(&mut buf),
                workspace.cast(),
                handle.cast(),
            )
        };
        success_or_panic(status);
    }
}

pub struct ConvBkwdDataGraph(NonNull<ConvBkwdDataDescriptor>);

impl ConvBkwdDataGraph {
    #[expect(clippy::too_many_arguments)]
    fn new<T>(
        x_shape: &[usize],
        x_stride: &[usize],
        w_shape: &[usize],
        w_stride: &[usize],
        y_shape: &[usize],
        y_stride: &[usize],
        pad: &[usize],
        stride: &[usize],
        dilation: &[usize],
    ) -> Self {
        let mut x_shape_stride = shape_stride_to_cudnn(x_shape, x_stride);
        let mut w_shape_stride = shape_stride_to_cudnn(w_shape, w_stride);
        let mut y_shape_stride = shape_stride_to_cudnn(y_shape, y_stride);
        let mut conv_info = get_conv_info(pad, stride, dilation);
        let data_type = get_cudnn_frontend_type::<T>();

        let mut inner = std::ptr::null_mut();
        let status = unsafe {
            zenu_cudnn_frontend_wrapper_sys::create_conv_backward_data_descriptor(
                std::ptr::from_mut(&mut inner),
                data_type,
                std::ptr::from_mut(&mut y_shape_stride),
                std::ptr::from_mut(&mut w_shape_stride),
                std::ptr::from_mut(&mut x_shape_stride),
                std::ptr::from_mut(&mut conv_info),
            )
        };
        success_or_panic(status);
        Self(NonNull::new(inner).unwrap())
    }

    fn check_graph(&self, handle: *mut cudnnHandle_t) {
        let status = unsafe {
            zenu_cudnn_frontend_wrapper_sys::check_conv_backward_data_graph(
                self.0.as_ptr(),
                handle.cast(),
            )
        };
        success_or_panic(status);
    }

    #[must_use]
    pub fn get_workspace_size(&self) -> usize {
        let mut size = 0;
        let status = unsafe {
            zenu_cudnn_frontend_wrapper_sys::get_conv_backward_data_workspace_size(
                self.0.as_ptr(),
                std::ptr::from_mut(&mut size),
            )
        };
        success_or_panic(status);
        size.try_into().unwrap()
    }

    pub fn execute<T>(
        &self,
        dy: *mut T,
        filter: *mut T,
        dx: *mut T,
        workspace: *mut T,
        handle: *mut cudnnHandle_t,
    ) {
        let mut buf = ConvBkwdDataBuffers {
            DY: dy.cast(),
            filter: filter.cast(),
            DX: dx.cast(),
        };
        let status = unsafe {
            zenu_cudnn_frontend_wrapper_sys::execute_conv_backward_data(
                self.0.as_ptr(),
                std::ptr::from_mut(&mut buf),
                workspace.cast(),
                handle.cast(),
            )
        };
        success_or_panic(status);
    }
}

pub struct ConvBkwdFilterGraph(NonNull<ConvBkwdFilterDescriptor>);

impl ConvBkwdFilterGraph {
    #[expect(clippy::too_many_arguments)]
    fn new<T>(
        x_shape: &[usize],
        x_stride: &[usize],
        w_shape: &[usize],
        w_stride: &[usize],
        y_shape: &[usize],
        y_stride: &[usize],
        pad: &[usize],
        stride: &[usize],
        dilation: &[usize],
    ) -> Self {
        let mut x_shape_stride = shape_stride_to_cudnn(x_shape, x_stride);
        let mut w_shape_stride = shape_stride_to_cudnn(w_shape, w_stride);
        let mut y_shape_stride = shape_stride_to_cudnn(y_shape, y_stride);
        let mut conv_info = get_conv_info(pad, stride, dilation);
        let data_type = get_cudnn_frontend_type::<T>();

        let mut inner = std::ptr::null_mut();
        let status = unsafe {
            zenu_cudnn_frontend_wrapper_sys::create_conv_backward_filter_descriptor(
                std::ptr::from_mut(&mut inner),
                data_type,
                std::ptr::from_mut(&mut y_shape_stride),
                std::ptr::from_mut(&mut x_shape_stride),
                std::ptr::from_mut(&mut w_shape_stride),
                std::ptr::from_mut(&mut conv_info),
            )
        };
        success_or_panic(status);
        Self(NonNull::new(inner).unwrap())
    }

    fn check_graph(&self, handle: *mut cudnnHandle_t) {
        let status = unsafe {
            zenu_cudnn_frontend_wrapper_sys::check_conv_backward_filter_graph(
                self.0.as_ptr(),
                handle.cast(),
            )
        };
        success_or_panic(status);
    }

    #[must_use]
    pub fn get_workspace_size(&self) -> usize {
        let mut size = 0;
        let status = unsafe {
            zenu_cudnn_frontend_wrapper_sys::get_conv_backward_filter_workspace_size(
                self.0.as_ptr(),
                std::ptr::from_mut(&mut size),
            )
        };
        success_or_panic(status);
        size.try_into().unwrap()
    }

    pub fn execute<T>(
        &self,
        dy: *mut T,
        x: *mut T,
        dw: *mut T,
        workspace: *mut T,
        handle: *mut cudnnHandle_t,
    ) {
        let mut buf = ConvBkwdFilterBuffers {
            X: x.cast(),
            DY: dy.cast(),
            DW: dw.cast(),
        };
        let status = unsafe {
            zenu_cudnn_frontend_wrapper_sys::execute_conv_backward_filter(
                self.0.as_ptr(),
                std::ptr::from_mut(&mut buf),
                workspace.cast(),
                handle.cast(),
            )
        };
        success_or_panic(status);
    }
}

pub struct ConvBuilder<T> {
    x_shape: Vec<usize>,
    w_shape: Vec<usize>,
    y_shape: Vec<usize>,
    x_stride: Vec<usize>,
    w_stride: Vec<usize>,
    y_stride: Vec<usize>,
    pad: Vec<usize>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for ConvBuilder<T> {
    fn default() -> Self {
        Self {
            x_shape: vec![],
            w_shape: vec![],
            y_shape: vec![],
            x_stride: vec![],
            w_stride: vec![],
            y_stride: vec![],
            pad: vec![],
            stride: vec![],
            dilation: vec![],
            _phantom: std::marker::PhantomData,
        }
    }
}

impl ConvBuilder<f32> {
    #[must_use]
    pub fn x_shape(mut self, x_shape: Vec<usize>) -> Self {
        self.x_shape = x_shape;
        self
    }

    #[must_use]
    pub fn w_shape(mut self, w_shape: Vec<usize>) -> Self {
        self.w_shape = w_shape;
        self
    }

    #[must_use]
    pub fn y_shape(mut self, y_shape: Vec<usize>) -> Self {
        self.y_shape = y_shape;
        self
    }

    #[must_use]
    pub fn x_stride(mut self, x_stride: Vec<usize>) -> Self {
        self.x_stride = x_stride;
        self
    }

    #[must_use]
    pub fn w_stride(mut self, w_stride: Vec<usize>) -> Self {
        self.w_stride = w_stride;
        self
    }

    #[must_use]
    pub fn y_stride(mut self, y_stride: Vec<usize>) -> Self {
        self.y_stride = y_stride;
        self
    }

    #[must_use]
    pub fn pad(mut self, pad: Vec<usize>) -> Self {
        self.pad = pad;
        self
    }

    #[must_use]
    pub fn stride(mut self, stride: Vec<usize>) -> Self {
        self.stride = stride;
        self
    }

    #[must_use]
    pub fn dilation(mut self, dilation: Vec<usize>) -> Self {
        self.dilation = dilation;
        self
    }

    fn is_build_able(&self) -> bool {
        !self.x_shape.is_empty()
            && !self.w_shape.is_empty()
            && !self.y_shape.is_empty()
            && !self.x_stride.is_empty()
            && !self.w_stride.is_empty()
            && !self.y_stride.is_empty()
            && !self.pad.is_empty()
            && !self.stride.is_empty()
            && !self.dilation.is_empty()
    }

    #[must_use]
    pub fn build_forward(self, handle: *mut cudnnHandle_t) -> ConvForwardGraph {
        self.is_build_able();
        let graph = ConvForwardGraph::new::<f32>(
            &self.x_shape,
            &self.x_stride,
            &self.w_shape,
            &self.w_stride,
            &self.y_shape,
            &self.y_stride,
            &self.pad,
            &self.stride,
            &self.dilation,
        );
        graph.check_graph(handle);
        graph
    }

    #[must_use]
    pub fn build_bkwd_data(self, handle: *mut cudnnHandle_t) -> ConvBkwdDataGraph {
        self.is_build_able();
        let graph = ConvBkwdDataGraph::new::<f32>(
            &self.x_shape,
            &self.x_stride,
            &self.w_shape,
            &self.w_stride,
            &self.y_shape,
            &self.y_stride,
            &self.pad,
            &self.stride,
            &self.dilation,
        );
        graph.check_graph(handle);
        graph
    }

    #[must_use]
    pub fn build_bkwd_filter(self, handle: *mut cudnnHandle_t) -> ConvBkwdFilterGraph {
        self.is_build_able();
        let graph = ConvBkwdFilterGraph::new::<f32>(
            &self.x_shape,
            &self.x_stride,
            &self.w_shape,
            &self.w_stride,
            &self.y_shape,
            &self.y_stride,
            &self.pad,
            &self.stride,
            &self.dilation,
        );
        graph.check_graph(handle);
        graph
    }
}

#[cfg(test)]
#[expect(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::cast_ptr_alignment
)]
mod graph_conv_test {
    use super::*;
    use crate::{
        runtime::{cuda_copy, cuda_malloc, ZenuCudaMemCopyKind},
        ZENU_CUDA_STATE,
    };

    #[expect(clippy::similar_names)]
    #[test]
    fn random() {
        let input = [
            0.5432947,
            -0.39515755,
            0.20552567,
            -0.45032975,
            -0.5730771,
            -0.5553584,
            0.59432304,
            1.5419426,
            1.8197253,
            -0.5515287,
            -1.325326,
            0.18855357,
            -0.069072686,
            -0.49492535,
            -1.4959149,
            -0.19383712,
            -0.4731198,
            0.33555076,
            1.5091219,
            2.0819554,
            1.7067116,
            2.3803675,
            -1.1256016,
            -0.3169981,
            -0.14067143,
            0.8057536,
            0.3276143,
            -0.7607072,
            -1.599082,
            0.018486667,
            -0.7504268,
            0.18540798,
        ];
        let output = [
            0.3671525,
            -0.17387724,
            -0.53952014,
            -0.41356063,
            0.13519445,
            -0.6369239,
            -0.5777169,
            -0.07820636,
            -0.6019154,
            -0.85000455,
            -0.227178,
            0.38553098,
            0.53258127,
            0.4952766,
            0.16334829,
            0.5179188,
            -1.1829954,
            -0.15092221,
            0.15374796,
            0.5376092,
            -0.35269666,
            -0.10102463,
            -0.628401,
            -0.40036133,
            -0.5694187,
            -0.1765114,
            -0.05552435,
            -0.3107502,
            -0.6736164,
            -0.44401115,
            -0.1804393,
            0.056986123,
            0.5652461,
            0.8913239,
            0.30458608,
            -0.7666081,
            0.15480474,
            0.14275207,
            0.42336845,
            0.12534592,
            0.5706087,
            0.40240055,
            -0.16282544,
            -0.032061294,
            0.47645676,
            -0.09869753,
            -0.34638345,
            -0.02880986,
        ];

        let filter = [
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

        let input_gpu = cuda_malloc::<f32>(input.len()).unwrap();
        cuda_copy(
            input_gpu,
            input.as_ptr(),
            input.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        let filter_gpu = cuda_malloc::<f32>(filter.len()).unwrap();
        cuda_copy(
            filter_gpu,
            filter.as_ptr(),
            filter.len(),
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();
        let output_gpu = cuda_malloc::<f32>(output.len()).unwrap();
        let context = ZENU_CUDA_STATE.lock().unwrap();
        let mut cudnn_handle: cudnnHandle_t = context.get_cudnn_handle();
        let conv_config = ConvBuilder::default()
            .x_shape(vec![1, 2, 4, 4])
            .x_stride(vec![2 * 4 * 4, 4 * 4, 4, 1])
            .w_shape(vec![3, 2, 3, 3])
            .w_stride(vec![2 * 3 * 3, 3 * 3, 3, 1])
            .y_shape(vec![1, 3, 4, 4])
            .y_stride(vec![3 * 4 * 4, 4 * 4, 4, 1])
            .pad(vec![1, 1])
            .stride(vec![1, 1])
            .dilation(vec![1, 1])
            .build_forward(std::ptr::from_mut(&mut cudnn_handle));
        let workspace_size = conv_config.get_workspace_size();
        let workspace_gpu = cuda_malloc::<u8>(workspace_size).unwrap();
        conv_config.execute(
            input_gpu,
            filter_gpu,
            output_gpu,
            workspace_gpu.cast(),
            std::ptr::from_mut(&mut cudnn_handle),
        );
        let mut output_cpu = vec![0.0; output.len()];
        cuda_copy(
            output_cpu.as_mut_ptr(),
            output_gpu,
            output.len(),
            ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();
        for (idx, (a, b)) in output_cpu.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "difference is too large {idx}, {}",
                a - b
            );
        }
    }
}

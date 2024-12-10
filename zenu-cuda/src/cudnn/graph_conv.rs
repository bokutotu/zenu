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

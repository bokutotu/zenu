use crate::{
    blas::Blas,
    dim::{Dim1, DimTrait},
    index::Index0D,
    matrix::{IndexAxisDyn, IndexItemAsign, MatrixBase, ToViewMatrix, ToViewMutMatrix, ViewMatrix},
    matrix_impl::{matrix_into_dim, Matrix},
    memory::{ToViewMemory, ToViewMutMemory, View},
    num::Num,
};

pub fn dot<T, X, Y>(x: X, y: Y) -> T
where
    T: Num,
    X: ViewMatrix + MatrixBase<Dim = Dim1, Item = T>,
    Y: ViewMatrix + MatrixBase<Dim = Dim1, Item = T>,
{
    dot_shape_check(x.shape(), y.shape()).unwrap();
    dot_unchecked(x, y)
}

pub(crate) fn dot_shape_check<XD, YD>(x_shape: XD, y_shape: YD) -> Result<(), String>
where
    XD: DimTrait,
    YD: DimTrait,
{
    if x_shape.len() != 1 || y_shape.len() != 1 {
        return Err("dot only supports 1-D arrays".to_string());
    }
    if x_shape[0] != y_shape[0] {
        return Err(format!(
            "shapes {:?} and {:?} not aligned: {:?} (dim 0) != {:?} (dim 0)",
            x_shape, y_shape, x_shape[0], y_shape[0]
        ));
    }
    Ok(())
}

pub(crate) fn dot_unchecked<T, X, Y>(x: X, y: Y) -> T
where
    T: Num,
    X: ViewMatrix + MatrixBase<Dim = Dim1, Item = T>,
    Y: ViewMatrix + MatrixBase<Dim = Dim1, Item = T>,
{
    X::Blas::dot(
        x.shape()[0],
        x.as_ptr(),
        x.stride()[0],
        y.as_ptr(),
        y.stride()[0],
    )
}

pub(crate) fn dot_batch_shape_check<SD, XD, YD>(
    self_shape: SD,
    x_shape: XD,
    y_shape: YD,
) -> Result<(), String>
where
    SD: DimTrait,
    XD: DimTrait,
    YD: DimTrait,
{
    if !(x_shape.len() == 2 || y_shape.len() == 2) {
        return Err("dot only supports 2-D arrays".to_string());
    }
    if self_shape.len() != 1 {
        return Err("dot only supports 1-D arrays(self_shape)".to_string());
    }
    if !((x_shape.len() == 2 && x_shape[1] == y_shape[0])
        || (y_shape.len() == 2 && x_shape[0] == y_shape[1]))
    {
        let x = if x_shape.len() == 2 {
            x_shape[1]
        } else {
            x_shape[0]
        };
        let y = if y_shape.len() == 2 {
            y_shape[0]
        } else {
            y_shape[1]
        };
        return Err(format!(
            "shapes {:?} and {:?} not aligned: {:?} (dim 1) != {:?} (dim 0)",
            x_shape, y_shape, x, y
        ));
    }
    if !((x_shape.len() == 2 || self_shape[0] == x_shape[0])
        || (y_shape.len() == 2 && self_shape[0] == y_shape[0]))
    {
        return Err(format!(
            "shapes {:?} and {:?} not aligned: {:?} (dim 0) != {:?} (dim 0)",
            self_shape, x_shape, self_shape[0], x_shape[0]
        ));
    }
    Ok(())
}

pub(crate) fn dot_batch_unchecked<T, SM, XM, YM, SD, XD, YD>(
    self_: Matrix<SM, SD>,
    x: Matrix<XM, XD>,
    y: Matrix<YM, YD>,
) where
    T: Num,
    SM: ToViewMutMemory<Item = T>,
    XM: ToViewMemory<Item = T>,
    YM: ToViewMemory<Item = T>,
    SD: DimTrait,
    XD: DimTrait,
    YD: DimTrait,
{
    let mut self_ = self_.into_dyn_dim();
    let x = x.into_dyn_dim();
    let y = y.into_dyn_dim();
    for i in 0..x.shape()[0] {
        let x = if x.shape().len() == 2 {
            x.index_axis_dyn(Index0D::new(i))
        } else {
            x.to_view()
        };
        let y = if y.shape().len() == 2 {
            y.index_axis_dyn(Index0D::new(i))
        } else {
            y.to_view()
        };
        let x = matrix_into_dim(x);
        let y = matrix_into_dim(y);
        let result = dot_unchecked(x, y);
        self_.to_view_mut().index_item_asign([i], result);
    }
}

pub fn dot_batch<T, SM, XM, YM, SD, XD, YD>(
    self_: Matrix<SM, SD>,
    x: Matrix<XM, XD>,
    y: Matrix<YM, YD>,
) where
    T: Num,
    SM: ToViewMutMemory<Item = T>,
    XM: View<Item = T>,
    YM: View<Item = T>,
    SD: DimTrait,
    XD: DimTrait,
    YD: DimTrait,
{
    dot_batch_shape_check(self_.shape(), x.shape(), y.shape()).unwrap();
    dot_batch_unchecked(self_, x, y);
}

use std::fmt;

use crate::{
    device::Device,
    dim::{DimDyn, DimTrait},
    index::Index0D,
    matrix::{Matrix, Ref, Repr},
    num::Num,
};
/// Default threshold, below this element count, we don't ellipsize
const ARRAY_MANY_ELEMENT_LIMIT: usize = 500;
/// Limit of element count for non-last axes before overflowing with an ellipsis.
const AXIS_LIMIT_STACKED: usize = 6;
/// Limit for next to last axis (printed as column)
/// An odd number because one element uses the same space as the ellipsis.
const AXIS_LIMIT_COL: usize = 11;
/// Limit for last axis (printed as row)
/// An odd number because one element uses approximately the space of the ellipsis.
const AXIS_LIMIT_ROW: usize = 11;

/// The string used as an ellipsis.
const ELLIPSIS: &str = "...";

#[derive(Clone, Debug)]
struct FormatOptions {
    axis_collapse_limit: usize,
    axis_collapse_limit_next_last: usize,
    axis_collapse_limit_last: usize,
}

impl FormatOptions {
    pub(crate) fn default_for_array(nelem: usize, no_limit: bool) -> Self {
        let default = Self {
            axis_collapse_limit: AXIS_LIMIT_STACKED,
            axis_collapse_limit_next_last: AXIS_LIMIT_COL,
            axis_collapse_limit_last: AXIS_LIMIT_ROW,
        };
        default.set_no_limit(no_limit || nelem < ARRAY_MANY_ELEMENT_LIMIT)
    }

    fn set_no_limit(mut self, no_limit: bool) -> Self {
        if no_limit {
            self.axis_collapse_limit = std::usize::MAX;
            self.axis_collapse_limit_next_last = std::usize::MAX;
            self.axis_collapse_limit_last = std::usize::MAX;
        }
        self
    }

    /// Axis length collapse limit before ellipsizing, where `axis_rindex` is
    /// the index of the axis from the back.
    pub(crate) fn collapse_limit(&self, axis_rindex: usize) -> usize {
        match axis_rindex {
            0 => self.axis_collapse_limit_last,
            1 => self.axis_collapse_limit_next_last,
            _ => self.axis_collapse_limit,
        }
    }
}

/// Formats the contents of a list of items, using an ellipsis to indicate when
/// the `length` of the list is greater than `limit`.
///
/// # Parameters
///
/// * `f`: The formatter.
/// * `length`: The length of the list.
/// * `limit`: The maximum number of items before overflow.
/// * `separator`: Separator to write between items.
/// * `ellipsis`: Ellipsis for indicating overflow.
/// * `fmt_elem`: A function that formats an element in the list, given the
///   formatter and the index of the item in the list.
fn format_with_overflow(
    f: &mut fmt::Formatter<'_>,
    length: usize,
    limit: usize,
    separator: &str,
    ellipsis: &str,
    fmt_elem: &mut dyn FnMut(&mut fmt::Formatter, usize) -> fmt::Result,
) -> fmt::Result {
    if length == 0 {
        // no-op
    } else if length <= limit {
        fmt_elem(f, 0)?;
        for i in 1..length {
            f.write_str(separator)?;
            fmt_elem(f, i)?
        }
    } else {
        let edge = limit / 2;
        fmt_elem(f, 0)?;
        for i in 1..edge {
            f.write_str(separator)?;
            fmt_elem(f, i)?;
        }
        f.write_str(separator)?;
        f.write_str(ellipsis)?;
        for i in length - edge..length {
            f.write_str(separator)?;
            fmt_elem(f, i)?
        }
    }
    Ok(())
}

fn format_array<A, R, S, D, F>(
    array: &Matrix<R, S, D>,
    f: &mut fmt::Formatter<'_>,
    format: F,
    fmt_opt: &FormatOptions,
) -> fmt::Result
where
    A: Num,
    F: FnMut(&A, &mut fmt::Formatter<'_>) -> fmt::Result + Clone,
    S: DimTrait,
    R: Repr<Item = A>,
    D: Device,
{
    // Cast into a dynamically dimensioned view
    // This is required to be able to use `index_axis` for the recursive case
    format_array_inner(
        array.to_ref().into_dyn_dim(),
        f,
        format,
        fmt_opt,
        0,
        array.shape().len(),
    )
}

fn format_array_inner<T, F, D>(
    view: Matrix<Ref<&T>, DimDyn, D>,
    f: &mut fmt::Formatter<'_>,
    mut format: F,
    fmt_opt: &FormatOptions,
    depth: usize,
    full_ndim: usize,
) -> fmt::Result
where
    T: Num,
    F: FnMut(&T, &mut fmt::Formatter<'_>) -> fmt::Result + Clone,
    D: Device,
{
    match view.shape().slice() {
        // If it's 0 dimensional, we just print out the scalar
        &[] => format(&view.index_item(&[] as &[usize]), f)?,
        // We handle 1-D arrays as a special case
        &[len] => {
            let view = view.into_dyn_dim();
            f.write_str("[")?;
            format_with_overflow(
                f,
                len,
                fmt_opt.collapse_limit(0),
                ", ",
                ELLIPSIS,
                &mut |f, index| format(&view.index_item([index]), f),
            )?;
            f.write_str("]")?;
        }
        // For n-dimensional arrays, we proceed recursively
        shape => {
            let blank_lines = "\n".repeat(shape.len() - 2);
            let indent = " ".repeat(depth + 1);
            let separator = format!(",\n{}{}", blank_lines, indent);

            f.write_str("[")?;
            let limit = fmt_opt.collapse_limit(full_ndim - depth - 1);
            format_with_overflow(f, shape[0], limit, &separator, ELLIPSIS, &mut |f, index| {
                format_array_inner(
                    view.index_axis_dyn(Index0D::new(index)),
                    f,
                    format.clone(),
                    fmt_opt,
                    depth + 1,
                    full_ndim,
                )
            })?;
            f.write_str("]")?;
        }
    }
    Ok(())
}

// NOTE: We can impl other fmt traits here
/// Format the array using `Display` and apply the formatting parameters used
/// to each element.
///
/// The array is shown in multiline style.
impl<A, R, S, D> fmt::Display for Matrix<R, S, D>
where
    A: Num + fmt::Display,
    R: Repr<Item = A>,
    S: DimTrait,
    D: Device,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fmt_opt = FormatOptions::default_for_array(self.shape().num_elm(), f.alternate());
        format_array(self, f, <_>::fmt, &fmt_opt)
    }
}

/// Format the array using `Debug` and apply the formatting parameters used
/// to each element.
///
/// The array is shown in multiline style.
impl<A, R, S, D> fmt::Debug for Matrix<R, S, D>
where
    A: Num + fmt::Debug,
    R: Repr<Item = A>,
    S: DimTrait,
    D: Device,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fmt_opt = FormatOptions::default_for_array(self.shape().num_elm(), f.alternate());
        format_array(self, f, <_>::fmt, &fmt_opt)?;

        // Add extra information for Debug
        write!(f, ", shape={:?}, strides={:?}", self.shape(), self.stride(),)?;
        // match self.shape().len() {
        //     Some(ndim) => write!(f, ", const ndim={}", ndim)?,
        //     None => write!(f, ", dynamic ndim={}", self.shape().len())?,
        // }
        Ok(())
    }
}

#[cfg(test)]
mod matrix_format {
    use crate::{
        device::Device,
        dim::{Dim2, Dim4, DimDyn},
        matrix::{Matrix, Owned},
    };

    fn assert_str_eq(expected: &str, actual: &str) {
        // use assert to avoid printing the strings twice on failure
        assert!(
            expected == actual,
            "formatting assertion failed\nexpected:\n{}\nactual:\n{}\n",
            expected,
            actual,
        );
    }

    fn small_array_1d<D: Device>() {
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1., 2., 3., 4., 5.], [5]);
        assert_eq!(format!("{}", a), "[1, 2, 3, 4, 5]");
    }
    #[test]
    fn small_array_1d_cpu() {
        small_array_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn small_array_1d_nvidia() {
        small_array_1d::<crate::device::nvidia::Nvidia>();
    }

    fn mid_array_1d<D: Device>() {
        let a: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.], [10]);
        assert_eq!(format!("{}", a), "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]");
    }
    #[test]
    fn mid_array_1d_cpu() {
        mid_array_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn mid_array_1d_nvidia() {
        mid_array_1d::<crate::device::nvidia::Nvidia>();
    }

    fn large_array_1d<D: Device>() {
        let mut v = Vec::new();
        for i in 1..=1000 {
            v.push(i as f32);
        }
        let a: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(v, [1000]);
        assert_eq!(
            format!("{}", a),
            "[1, 2, 3, 4, 5, ..., 996, 997, 998, 999, 1000]"
        );
    }
    #[test]
    fn large_array_1d_cpu() {
        large_array_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn large_array_1d_nvidia() {
        large_array_1d::<crate::device::nvidia::Nvidia>();
    }

    fn dim_2_last_axis_overflow<D: Device>() {
        let a: Matrix<Owned<f32>, Dim2, D> = Matrix::ones([22, 24]);
        let actual = format!("{}", a);
        let expected = "\
[[1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
 ...,
 [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1]]";
        assert_str_eq(expected, &actual);
    }
    #[test]
    fn dim_2_last_axis_overflow_cpu() {
        dim_2_last_axis_overflow::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn dim_2_last_axis_overflow_nvidia() {
        dim_2_last_axis_overflow::<crate::device::nvidia::Nvidia>();
    }

    fn dim_3_overflow_most<D: Device>() {
        let mut v = Vec::new();
        for i in 0..7 {
            for j in 0..11 {
                for k in 0..12 {
                    v.push(
                        1000.
                            + (100. * ((i as f64).sqrt() + (j as f64).sin() + k as f64)).round()
                                / 100.,
                    );
                }
            }
        }
        // let a = Array3::from_shape_fn(
        //     (AXIS_LIMIT_STACKED + 1, AXIS_LIMIT_COL, AXIS_LIMIT_ROW + 1),
        //     |(i, j, k)| {
        //         1000. + (100. * ((i as f64).sqrt() + (j as f64).sin() + k as f64)).round() / 100.
        //     },
        // );
        let a: Matrix<Owned<f64>, DimDyn, D> = Matrix::from_vec(v, [7, 11, 12]);
        let actual = format!("{:6.1}", a);
        let expected = "\
[[[1000.0, 1001.0, 1002.0, 1003.0, 1004.0, ..., 1007.0, 1008.0, 1009.0, 1010.0, 1011.0],
  [1000.8, 1001.8, 1002.8, 1003.8, 1004.8, ..., 1007.8, 1008.8, 1009.8, 1010.8, 1011.8],
  [1000.9, 1001.9, 1002.9, 1003.9, 1004.9, ..., 1007.9, 1008.9, 1009.9, 1010.9, 1011.9],
  [1000.1, 1001.1, 1002.1, 1003.1, 1004.1, ..., 1007.1, 1008.1, 1009.1, 1010.1, 1011.1],
  [ 999.2, 1000.2, 1001.2, 1002.2, 1003.2, ..., 1006.2, 1007.2, 1008.2, 1009.2, 1010.2],
  [ 999.0, 1000.0, 1001.0, 1002.0, 1003.0, ..., 1006.0, 1007.0, 1008.0, 1009.0, 1010.0],
  [ 999.7, 1000.7, 1001.7, 1002.7, 1003.7, ..., 1006.7, 1007.7, 1008.7, 1009.7, 1010.7],
  [1000.7, 1001.7, 1002.7, 1003.7, 1004.7, ..., 1007.7, 1008.7, 1009.7, 1010.7, 1011.7],
  [1001.0, 1002.0, 1003.0, 1004.0, 1005.0, ..., 1008.0, 1009.0, 1010.0, 1011.0, 1012.0],
  [1000.4, 1001.4, 1002.4, 1003.4, 1004.4, ..., 1007.4, 1008.4, 1009.4, 1010.4, 1011.4],
  [ 999.5, 1000.5, 1001.5, 1002.5, 1003.5, ..., 1006.5, 1007.5, 1008.5, 1009.5, 1010.5]],

 [[1001.0, 1002.0, 1003.0, 1004.0, 1005.0, ..., 1008.0, 1009.0, 1010.0, 1011.0, 1012.0],
  [1001.8, 1002.8, 1003.8, 1004.8, 1005.8, ..., 1008.8, 1009.8, 1010.8, 1011.8, 1012.8],
  [1001.9, 1002.9, 1003.9, 1004.9, 1005.9, ..., 1008.9, 1009.9, 1010.9, 1011.9, 1012.9],
  [1001.1, 1002.1, 1003.1, 1004.1, 1005.1, ..., 1008.1, 1009.1, 1010.1, 1011.1, 1012.1],
  [1000.2, 1001.2, 1002.2, 1003.2, 1004.2, ..., 1007.2, 1008.2, 1009.2, 1010.2, 1011.2],
  [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, ..., 1007.0, 1008.0, 1009.0, 1010.0, 1011.0],
  [1000.7, 1001.7, 1002.7, 1003.7, 1004.7, ..., 1007.7, 1008.7, 1009.7, 1010.7, 1011.7],
  [1001.7, 1002.7, 1003.7, 1004.7, 1005.7, ..., 1008.7, 1009.7, 1010.7, 1011.7, 1012.7],
  [1002.0, 1003.0, 1004.0, 1005.0, 1006.0, ..., 1009.0, 1010.0, 1011.0, 1012.0, 1013.0],
  [1001.4, 1002.4, 1003.4, 1004.4, 1005.4, ..., 1008.4, 1009.4, 1010.4, 1011.4, 1012.4],
  [1000.5, 1001.5, 1002.5, 1003.5, 1004.5, ..., 1007.5, 1008.5, 1009.5, 1010.5, 1011.5]],

 [[1001.4, 1002.4, 1003.4, 1004.4, 1005.4, ..., 1008.4, 1009.4, 1010.4, 1011.4, 1012.4],
  [1002.3, 1003.3, 1004.3, 1005.3, 1006.3, ..., 1009.3, 1010.3, 1011.3, 1012.3, 1013.3],
  [1002.3, 1003.3, 1004.3, 1005.3, 1006.3, ..., 1009.3, 1010.3, 1011.3, 1012.3, 1013.3],
  [1001.6, 1002.6, 1003.6, 1004.6, 1005.6, ..., 1008.6, 1009.6, 1010.6, 1011.6, 1012.6],
  [1000.7, 1001.7, 1002.7, 1003.7, 1004.7, ..., 1007.7, 1008.7, 1009.7, 1010.7, 1011.7],
  [1000.5, 1001.5, 1002.5, 1003.5, 1004.5, ..., 1007.5, 1008.5, 1009.5, 1010.5, 1011.5],
  [1001.1, 1002.1, 1003.1, 1004.1, 1005.1, ..., 1008.1, 1009.1, 1010.1, 1011.1, 1012.1],
  [1002.1, 1003.1, 1004.1, 1005.1, 1006.1, ..., 1009.1, 1010.1, 1011.1, 1012.1, 1013.1],
  [1002.4, 1003.4, 1004.4, 1005.4, 1006.4, ..., 1009.4, 1010.4, 1011.4, 1012.4, 1013.4],
  [1001.8, 1002.8, 1003.8, 1004.8, 1005.8, ..., 1008.8, 1009.8, 1010.8, 1011.8, 1012.8],
  [1000.9, 1001.9, 1002.9, 1003.9, 1004.9, ..., 1007.9, 1008.9, 1009.9, 1010.9, 1011.9]],

 ...,

 [[1002.0, 1003.0, 1004.0, 1005.0, 1006.0, ..., 1009.0, 1010.0, 1011.0, 1012.0, 1013.0],
  [1002.8, 1003.8, 1004.8, 1005.8, 1006.8, ..., 1009.8, 1010.8, 1011.8, 1012.8, 1013.8],
  [1002.9, 1003.9, 1004.9, 1005.9, 1006.9, ..., 1009.9, 1010.9, 1011.9, 1012.9, 1013.9],
  [1002.1, 1003.1, 1004.1, 1005.1, 1006.1, ..., 1009.1, 1010.1, 1011.1, 1012.1, 1013.1],
  [1001.2, 1002.2, 1003.2, 1004.2, 1005.2, ..., 1008.2, 1009.2, 1010.2, 1011.2, 1012.2],
  [1001.0, 1002.0, 1003.0, 1004.0, 1005.0, ..., 1008.0, 1009.0, 1010.0, 1011.0, 1012.0],
  [1001.7, 1002.7, 1003.7, 1004.7, 1005.7, ..., 1008.7, 1009.7, 1010.7, 1011.7, 1012.7],
  [1002.7, 1003.7, 1004.7, 1005.7, 1006.7, ..., 1009.7, 1010.7, 1011.7, 1012.7, 1013.7],
  [1003.0, 1004.0, 1005.0, 1006.0, 1007.0, ..., 1010.0, 1011.0, 1012.0, 1013.0, 1014.0],
  [1002.4, 1003.4, 1004.4, 1005.4, 1006.4, ..., 1009.4, 1010.4, 1011.4, 1012.4, 1013.4],
  [1001.5, 1002.5, 1003.5, 1004.5, 1005.5, ..., 1008.5, 1009.5, 1010.5, 1011.5, 1012.5]],

 [[1002.2, 1003.2, 1004.2, 1005.2, 1006.2, ..., 1009.2, 1010.2, 1011.2, 1012.2, 1013.2],
  [1003.1, 1004.1, 1005.1, 1006.1, 1007.1, ..., 1010.1, 1011.1, 1012.1, 1013.1, 1014.1],
  [1003.1, 1004.1, 1005.1, 1006.1, 1007.1, ..., 1010.1, 1011.1, 1012.1, 1013.1, 1014.1],
  [1002.4, 1003.4, 1004.4, 1005.4, 1006.4, ..., 1009.4, 1010.4, 1011.4, 1012.4, 1013.4],
  [1001.5, 1002.5, 1003.5, 1004.5, 1005.5, ..., 1008.5, 1009.5, 1010.5, 1011.5, 1012.5],
  [1001.3, 1002.3, 1003.3, 1004.3, 1005.3, ..., 1008.3, 1009.3, 1010.3, 1011.3, 1012.3],
  [1002.0, 1003.0, 1004.0, 1005.0, 1006.0, ..., 1009.0, 1010.0, 1011.0, 1012.0, 1013.0],
  [1002.9, 1003.9, 1004.9, 1005.9, 1006.9, ..., 1009.9, 1010.9, 1011.9, 1012.9, 1013.9],
  [1003.2, 1004.2, 1005.2, 1006.2, 1007.2, ..., 1010.2, 1011.2, 1012.2, 1013.2, 1014.2],
  [1002.6, 1003.6, 1004.6, 1005.6, 1006.6, ..., 1009.6, 1010.6, 1011.6, 1012.6, 1013.6],
  [1001.7, 1002.7, 1003.7, 1004.7, 1005.7, ..., 1008.7, 1009.7, 1010.7, 1011.7, 1012.7]],

 [[1002.5, 1003.5, 1004.5, 1005.5, 1006.5, ..., 1009.5, 1010.5, 1011.5, 1012.5, 1013.5],
  [1003.3, 1004.3, 1005.3, 1006.3, 1007.3, ..., 1010.3, 1011.3, 1012.3, 1013.3, 1014.3],
  [1003.4, 1004.4, 1005.4, 1006.4, 1007.4, ..., 1010.4, 1011.4, 1012.4, 1013.4, 1014.4],
  [1002.6, 1003.6, 1004.6, 1005.6, 1006.6, ..., 1009.6, 1010.6, 1011.6, 1012.6, 1013.6],
  [1001.7, 1002.7, 1003.7, 1004.7, 1005.7, ..., 1008.7, 1009.7, 1010.7, 1011.7, 1012.7],
  [1001.5, 1002.5, 1003.5, 1004.5, 1005.5, ..., 1008.5, 1009.5, 1010.5, 1011.5, 1012.5],
  [1002.2, 1003.2, 1004.2, 1005.2, 1006.2, ..., 1009.2, 1010.2, 1011.2, 1012.2, 1013.2],
  [1003.1, 1004.1, 1005.1, 1006.1, 1007.1, ..., 1010.1, 1011.1, 1012.1, 1013.1, 1014.1],
  [1003.4, 1004.4, 1005.4, 1006.4, 1007.4, ..., 1010.4, 1011.4, 1012.4, 1013.4, 1014.4],
  [1002.9, 1003.9, 1004.9, 1005.9, 1006.9, ..., 1009.9, 1010.9, 1011.9, 1012.9, 1013.9],
  [1001.9, 1002.9, 1003.9, 1004.9, 1005.9, ..., 1008.9, 1009.9, 1010.9, 1011.9, 1012.9]]]";
        assert_str_eq(expected, &actual);
    }
    #[test]
    fn dim_3_overflow_most_cpu() {
        dim_3_overflow_most::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn dim_3_overflow_most_nvidia() {
        dim_3_overflow_most::<crate::device::nvidia::Nvidia>();
    }

    fn dim_4_overflow_outer<D: Device>() {
        // let a = Array4::from_shape_fn((10, 10, 3, 3), |(i, j, k, l)| i + j + k + l);
        let mut v = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..3 {
                    for l in 0..3 {
                        v.push((i + j + k + l) as f32);
                    }
                }
            }
        }
        let a: Matrix<Owned<f32>, Dim4, D> = Matrix::from_vec(v, [10, 10, 3, 3]);
        let actual = format!("{:2}", a);
        // Generated using NumPy with:
        // np.set_printoptions(threshold=500, edgeitems=3)
        // np.fromfunction(lambda i, j, k, l: i + j + k + l, (10, 10, 3, 3), dtype=int)
        //
        let expected = "\
[[[[ 0,  1,  2],
   [ 1,  2,  3],
   [ 2,  3,  4]],

  [[ 1,  2,  3],
   [ 2,  3,  4],
   [ 3,  4,  5]],

  [[ 2,  3,  4],
   [ 3,  4,  5],
   [ 4,  5,  6]],

  ...,

  [[ 7,  8,  9],
   [ 8,  9, 10],
   [ 9, 10, 11]],

  [[ 8,  9, 10],
   [ 9, 10, 11],
   [10, 11, 12]],

  [[ 9, 10, 11],
   [10, 11, 12],
   [11, 12, 13]]],


 [[[ 1,  2,  3],
   [ 2,  3,  4],
   [ 3,  4,  5]],

  [[ 2,  3,  4],
   [ 3,  4,  5],
   [ 4,  5,  6]],

  [[ 3,  4,  5],
   [ 4,  5,  6],
   [ 5,  6,  7]],

  ...,

  [[ 8,  9, 10],
   [ 9, 10, 11],
   [10, 11, 12]],

  [[ 9, 10, 11],
   [10, 11, 12],
   [11, 12, 13]],

  [[10, 11, 12],
   [11, 12, 13],
   [12, 13, 14]]],


 [[[ 2,  3,  4],
   [ 3,  4,  5],
   [ 4,  5,  6]],

  [[ 3,  4,  5],
   [ 4,  5,  6],
   [ 5,  6,  7]],

  [[ 4,  5,  6],
   [ 5,  6,  7],
   [ 6,  7,  8]],

  ...,

  [[ 9, 10, 11],
   [10, 11, 12],
   [11, 12, 13]],

  [[10, 11, 12],
   [11, 12, 13],
   [12, 13, 14]],

  [[11, 12, 13],
   [12, 13, 14],
   [13, 14, 15]]],


 ...,


 [[[ 7,  8,  9],
   [ 8,  9, 10],
   [ 9, 10, 11]],

  [[ 8,  9, 10],
   [ 9, 10, 11],
   [10, 11, 12]],

  [[ 9, 10, 11],
   [10, 11, 12],
   [11, 12, 13]],

  ...,

  [[14, 15, 16],
   [15, 16, 17],
   [16, 17, 18]],

  [[15, 16, 17],
   [16, 17, 18],
   [17, 18, 19]],

  [[16, 17, 18],
   [17, 18, 19],
   [18, 19, 20]]],


 [[[ 8,  9, 10],
   [ 9, 10, 11],
   [10, 11, 12]],

  [[ 9, 10, 11],
   [10, 11, 12],
   [11, 12, 13]],

  [[10, 11, 12],
   [11, 12, 13],
   [12, 13, 14]],

  ...,

  [[15, 16, 17],
   [16, 17, 18],
   [17, 18, 19]],

  [[16, 17, 18],
   [17, 18, 19],
   [18, 19, 20]],

  [[17, 18, 19],
   [18, 19, 20],
   [19, 20, 21]]],


 [[[ 9, 10, 11],
   [10, 11, 12],
   [11, 12, 13]],

  [[10, 11, 12],
   [11, 12, 13],
   [12, 13, 14]],

  [[11, 12, 13],
   [12, 13, 14],
   [13, 14, 15]],

  ...,

  [[16, 17, 18],
   [17, 18, 19],
   [18, 19, 20]],

  [[17, 18, 19],
   [18, 19, 20],
   [19, 20, 21]],

  [[18, 19, 20],
   [19, 20, 21],
   [20, 21, 22]]]]";
        assert_str_eq(expected, &actual);
    }
    #[test]
    fn dim_4_overflow_outer_cpu() {
        dim_4_overflow_outer::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn dim_4_overflow_outer_nvidia() {
        dim_4_overflow_outer::<crate::device::nvidia::Nvidia>();
    }
}

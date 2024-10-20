use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

#[derive(Clone, Debug, Copy, PartialEq, Default)]
pub struct SliceDim {
    pub(crate) start: Option<usize>,
    pub(crate) end: Option<usize>,
    pub(crate) step: Option<usize>,
}

impl SliceDim {
    #[must_use]
    pub fn step(self, step: usize) -> Self {
        Self {
            start: self.start,
            end: self.end,
            step: Some(step),
        }
    }

    fn validate(&self, dim: usize) -> bool {
        let start = self.start.unwrap_or(0);
        let end = self.end.unwrap_or(dim - 1);
        let step = self.step.unwrap_or(1);

        if start > end {
            return false;
        }

        if start > dim {
            return false;
        }

        if step == 0 {
            return false;
        }

        true
    }

    fn new_dim_unchanged(&self, dim: usize) -> usize {
        let start = self.start.unwrap_or(0);
        let mut end = self.end.unwrap_or(dim);
        let step = self.step.unwrap_or(1);

        if end > dim {
            end = dim;
        }

        (end - start + step - 1) / step
    }

    pub(super) fn new_dim(&self, dim: usize) -> usize {
        if self.validate(dim) {
            return self.new_dim_unchanged(dim);
        }
        panic!("invalid slice");
    }

    pub(super) fn new_stride(&self, stride: usize) -> usize {
        let step = self.step.unwrap_or(1);
        stride * step
    }
}

impl From<Range<usize>> for SliceDim {
    fn from(range: Range<usize>) -> Self {
        SliceDim {
            start: Some(range.start),
            end: Some(range.end),
            step: None,
        }
    }
}

impl From<RangeFull> for SliceDim {
    fn from(_: RangeFull) -> Self {
        SliceDim {
            start: None,
            end: None,
            step: None,
        }
    }
}

impl From<RangeTo<usize>> for SliceDim {
    fn from(range: RangeTo<usize>) -> Self {
        SliceDim {
            start: None,
            end: Some(range.end),
            step: None,
        }
    }
}

impl From<RangeFrom<usize>> for SliceDim {
    fn from(range: RangeFrom<usize>) -> Self {
        SliceDim {
            start: Some(range.start),
            end: None,
            step: None,
        }
    }
}

impl From<RangeInclusive<usize>> for SliceDim {
    fn from(range: RangeInclusive<usize>) -> Self {
        SliceDim {
            start: Some(*range.start()),
            end: Some(*range.end()),
            step: None,
        }
    }
}

impl From<RangeToInclusive<usize>> for SliceDim {
    fn from(range: RangeToInclusive<usize>) -> Self {
        SliceDim {
            start: None,
            end: Some(range.end + 1),
            step: None,
        }
    }
}

impl From<usize> for SliceDim {
    fn from(index: usize) -> Self {
        SliceDim {
            start: Some(index),
            end: Some(index),
            step: None,
        }
    }
}

#[test]
fn slice_index() {
    let slice_dim = SliceDim {
        start: Some(0),
        end: Some(10),
        step: None,
    };

    let dim = 20;
    let new_dim = slice_dim.new_dim(dim);
    assert_eq!(new_dim, 10);
    let new_stride = slice_dim.new_stride(1);
    assert_eq!(new_stride, 1);
}

#[test]
fn slice_index_with_stride() {
    let slice_dim = SliceDim {
        start: Some(0),
        end: Some(10),
        step: Some(2),
    };

    let dim = 20;
    let new_dim = slice_dim.new_dim(dim);
    assert_eq!(new_dim, 5);
    let new_stride = slice_dim.new_stride(1);
    assert_eq!(new_stride, 2);
}

#[test]
fn slice_dim_full_range() {
    let slice_dim = SliceDim {
        start: None,
        end: None,
        step: None,
    };

    let dim = 20;
    let new_dim = slice_dim.new_dim(dim);
    assert_eq!(new_dim, 20);
    let new_stride = slice_dim.new_stride(1);
    assert_eq!(new_stride, 1);
}

#[macro_export]
macro_rules! slice(
    (@parse [$($stack:tt)*] $r:expr;$s:expr) => {
        $crate::slice!(@final
            [$($stack)* $crate::slice!(@convert $r, $s)]
        )
    };
    (@parse [$($stack:tt)*] $r:expr) => {
        $crate::slice!(@final [$($stack)* $crate::slice!(@convert $r)])
    };
    (@parse [$($stack:tt)*] $r:expr;$s:expr ,) => {
        $crate::slice![@parse $in_dim, $out_dim, [$($stack)*] $r;$s]
    };
    (@parse $in_dim:expr, $out_dim:expr, [$($stack:tt)*] $r:expr ,) => {
        $crate::slice![@parse $in_dim, $out_dim, [$($stack)*] $r]
    };
    (@parse [$($stack:tt)*] $r:expr;$s:expr, $($t:tt)*) => {
        match $r {
            r => {
                $crate::slice![@parse
                   [$($stack)* $crate::slice!(@convert r, $s),]
                   $($t)*
                ]
            }
        }
    };
    (@parse [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        match $r {
            r => {
                $crate::slice![@parse
                   [$($stack)* $crate::slice!(@convert r),]
                   $($t)*
                ]
            }
        }
    };
    (@parse []) => {
        {
            Slice0D {  }
        }
    };
    (@parse $($t:tt)*) => { compile_error!("Invalid syntax in slice![] call.") };

    (@convert $r:expr) => {
        $crate::slice_impl::SliceDim::from($r)
    };
    (@convert $r:expr, $s:expr) => {
        $crate::slice_impl::SliceDim::from($r).step($s)
    };

    (@final [$dim1:expr]) => {
        match $dim1 {
            dim1 => $crate::slice_impl::Slice1D {
                    index: [dim1],
                }
        }
    };

    (@final [$dim1:expr, $dim2:expr]) => {
        $crate::slice_impl::Slice2D {
            index: [$dim1, $dim2],
        }
    };

    (@final [$dim1:expr, $dim2:expr, $dim3:expr]) => {
        $crate::slice_impl::Slice3D {
            index: [$dim1, $dim2, $dim3],
        }
    };

    (@final [$dim1:expr, $dim2:expr, $dim3:expr, $dim4:expr]) => {
        $crate::slice_impl::Slice4D {
            index: [$dim1, $dim2, $dim3, $dim4],
        }
    };

    ($($t:tt)*) => {
        $crate::slice![@parse
              []
              $($t)*
        ]
    };
);

#[cfg(test)]
mod tests {
    use crate::slice_impl::{Slice1D, Slice2D, Slice3D, Slice4D, SliceDim};

    #[test]
    fn test_slice_macro_1d() {
        let slice = slice!(1..5;2);
        assert_eq!(
            slice,
            Slice1D {
                index: [SliceDim {
                    start: Some(1),
                    end: Some(5),
                    step: Some(2)
                }]
            }
        );
    }

    #[test]
    fn test_slice_macro_2d() {
        let slice = slice!(..;2, 3..);
        assert_eq!(
            slice,
            Slice2D {
                index: [
                    SliceDim {
                        start: None,
                        end: None,
                        step: Some(2)
                    },
                    SliceDim {
                        start: Some(3),
                        end: None,
                        step: None
                    }
                ]
            }
        );
    }

    #[test]
    fn test_slice_macro_3d() {
        let slice = slice!(1..3;1, ..4;2, 5..;3);
        assert_eq!(
            slice,
            Slice3D {
                index: [
                    SliceDim {
                        start: Some(1),
                        end: Some(3),
                        step: Some(1)
                    },
                    SliceDim {
                        start: None,
                        end: Some(4),
                        step: Some(2)
                    },
                    SliceDim {
                        start: Some(5),
                        end: None,
                        step: Some(3)
                    }
                ]
            }
        );
    }

    #[test]
    fn test_slice_macro_4d() {
        let slice = slice!(..;1, 2..5, ..6;2, 7..9;3);
        assert_eq!(
            slice,
            Slice4D {
                index: [
                    SliceDim {
                        start: None,
                        end: None,
                        step: Some(1)
                    },
                    SliceDim {
                        start: Some(2),
                        end: Some(5),
                        step: None
                    },
                    SliceDim {
                        start: None,
                        end: Some(6),
                        step: Some(2)
                    },
                    SliceDim {
                        start: Some(7),
                        end: Some(9),
                        step: Some(3)
                    }
                ]
            }
        );
    }
}

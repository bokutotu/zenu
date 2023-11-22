// ペアノの自然数
pub trait Nat {}
pub struct Zero;
pub struct Succ<N: Nat>(N);

impl Nat for Zero {}
impl<N: Nat> Nat for Succ<N> {}

#[macro_export]
macro_rules! to_nat {
    (0) => { Zero };
    (1) => { Succ<Zero> };
    (2) => { Succ<Succ<Zero>> };
    (3) => { Succ<Succ<Succ<Zero>>> };
    (4) => { Succ<Succ<Succ<Succ<Zero>>>> };
    // 以降、必要な数まで続ける...
}

// IndexDim の定義
#[derive(Clone, Debug, Copy)]
enum IndexDim {
    Index(isize),
    Range {
        start: Option<isize>,
        end: Option<isize>,
        step: Option<isize>,
    },
}

// IndexND 構造体の定義
#[derive(Clone, Debug, Copy)]
struct Index1D<N: Nat> {
    index: IndexDim,
    _marker: std::marker::PhantomData<N>,
}

#[derive(Clone, Debug, Copy)]
struct Index2D<N: Nat> {
    index: [IndexDim; 2],
    _marker: std::marker::PhantomData<N>,
}

#[derive(Clone, Debug, Copy)]
struct Index3D<N: Nat> {
    index: [IndexDim; 3],
    _marker: std::marker::PhantomData<N>,
}

#[derive(Clone, Debug, Copy)]
struct Index4D<N: Nat> {
    index: [IndexDim; 4],
    _marker: std::marker::PhantomData<N>,
}

#[macro_export]
macro_rules! index_dim {
    (..) => {
        IndexDim::Range {
            start: None,
            end: None,
            step: None,
        }
    };
    ($start:tt) => {
        IndexDim::Index($start)
    };
    ($start:tt..) => {
        IndexDim::Range {
            start: Some($start),
            end: None,
            step: None,
        }
    };
    (..$end:tt) => {
        IndexDim::Range {
            start: None,
            end: Some($end),
            step: None,
        }
    };
    (..;$step:tt) => {
        IndexDim::Range {
            start: None,
            end: None,
            step: Some($step),
        }
    };
    ($start:tt..$end:tt) => {
        IndexDim::Range {
            start: Some($start),
            end: Some($end),
            step: None,
        }
    };
    ($start:tt..;$step:tt) => {
        IndexDim::Range {
            start: Some($start),
            end: None,
            step: Some($step),
        }
    };
    (..$end:tt;$step:tt) => {
        IndexDim::Range {
            start: None,
            end: Some($end),
            step: Some($step),
        }
    };
    ($start:tt..$end:tt;$step:tt) => {
        IndexDim::Range {
            start: Some($start),
            end: Some($end),
            step: Some($step),
        }
    };
}

#[macro_export]
macro_rules! index {
    ($($dim:tt),*) => {
        {
            let dims = [$(index_dim!($dim)),*];
            let index_count = dims.iter().filter(|d| matches!(d, IndexDim::Index(_))).count();
            match dims.len() {
                1 => Index1D::<{ crate::to_nat!(index_count) }> { index: dims[0], _marker: std::marker::PhantomData },
                2 => Index2D::<{ crate::to_nat!(index_count) }> { index: dims, _marker: std::marker::PhantomData },
                3 => Index3D::<{ crate::to_nat!(index_count) }> { index: dims, _marker: std::marker::PhantomData },
                4 => Index4D::<{ crate::to_nat!(index_count) }> { index: dims, _marker: std::marker::PhantomData },
                _ => panic!("Unsupported number of dimensions"),
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // IndexDim::Index のテスト
    #[test]
    fn test_index_only() {
        let index = index_dim!(1);
        match index {
            IndexDim::Index(val) => assert_eq!(val, 1),
            _ => panic!("Expected IndexDim::Index"),
        }
    }

    // IndexDim::Range の全8パターンのテスト
    // start のみ
    #[test]
    fn test_range_start_only() {
        let index = index_dim!(1..);
        match index {
            IndexDim::Range { start, end, step } => {
                assert_eq!(start, Some(1));
                assert!(end.is_none());
                assert!(step.is_none());
            }
            _ => panic!("Expected IndexDim::Range with start only"),
        }
    }

    // end のみ
    #[test]
    fn test_range_end_only() {
        let index = index_dim!(..2);
        match index {
            IndexDim::Range { start, end, step } => {
                assert!(start.is_none());
                assert_eq!(end, Some(2));
                assert!(step.is_none());
            }
            _ => panic!("Expected IndexDim::Range with end only"),
        }
    }

    // step のみ
    #[test]
    fn test_range_step_only() {
        let index = index_dim!(..;3);
        match index {
            IndexDim::Range { start, end, step } => {
                assert!(start.is_none());
                assert!(end.is_none());
                assert_eq!(step, Some(3));
            }
            _ => panic!("Expected IndexDim::Range with step only"),
        }
    }

    // start と end
    #[test]
    fn test_range_start_end() {
        let index = index_dim!(1..2);
        match index {
            IndexDim::Range { start, end, step } => {
                assert_eq!(start, Some(1));
                assert_eq!(end, Some(2));
                assert!(step.is_none());
            }
            _ => panic!("Expected IndexDim::Range with start and end"),
        }
    }

    // start と step
    #[test]
    fn test_range_start_step() {
        let index = index_dim!(1..;3);
        match index {
            IndexDim::Range { start, end, step } => {
                assert_eq!(start, Some(1));
                assert!(end.is_none());
                assert_eq!(step, Some(3));
            }
            _ => panic!("Expected IndexDim::Range with start and step"),
        }
    }

    // end と step
    #[test]
    fn test_range_end_step() {
        let index = index_dim!(..2;3);
        match index {
            IndexDim::Range { start, end, step } => {
                assert!(start.is_none());
                assert_eq!(end, Some(2));
                assert_eq!(step, Some(3));
            }
            _ => panic!("Expected IndexDim::Range with end and step"),
        }
    }

    // start、end、step
    #[test]
    fn test_range_start_end_step() {
        let index = index_dim!(1..2;3);
        match index {
            IndexDim::Range { start, end, step } => {
                assert_eq!(start, Some(1));
                assert_eq!(end, Some(2));
                assert_eq!(step, Some(3));
            }
            _ => panic!("Expected IndexDim::Range with start, end, and step"),
        }
    }

    // 全て省略
    #[test]
    fn test_range_none() {
        let index = index_dim!(..);
        match index {
            IndexDim::Range { start, end, step } => {
                assert!(start.is_none());
                assert!(end.is_none());
                assert!(step.is_none());
            }
            _ => panic!("Expected IndexDim::Range with all None"),
        }
    }
}

use std::ops::{Index, IndexMut};

use crate::dim::{DimTrait, GreaterDimTrait, LessDimTrait};

#[derive(Clone, Debug, Copy, Default)]
pub struct Dim0 {}

impl Dim0 {
    pub fn new() -> Self {
        Self {}
    }
}

impl Index<usize> for Dim0 {
    type Output = usize;

    fn index(&self, _: usize) -> &Self::Output {
        &0
    }
}

impl IndexMut<usize> for Dim0 {
    fn index_mut(&mut self, _: usize) -> &mut Self::Output {
        todo!();
    }
}

impl PartialEq for Dim0 {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Iterator for Dim0 {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl DimTrait for Dim0 {
    fn len(&self) -> usize {
        0
    }

    fn is_empty(&self) -> bool {
        todo!();
    }

    fn is_overflow<D: DimTrait>(&self, index: D) -> bool {
        // if index._len() == 0 {
        //     false
        // } else {
        //     true
        // }
        index.len() != 0
    }
}

// impl IndexTrait for Dim0 {
//     type Dim = Self;
//     fn offset(&self, _: &Self::Dim, _: &Self::Dim) -> usize {
//         0
//     }
// }

macro_rules! impl_dim {
    ($name:ident, $index_ty:ty) => {
        #[derive(Clone, Debug, Copy, Default)]
        pub struct $name {
            dim: $index_ty,
        }

        impl $name {
            pub fn new(dim: $index_ty) -> Self {
                Self { dim }
            }

            pub fn dim(&self) -> $index_ty {
                self.dim
            }
        }

        impl Index<usize> for $name {
            type Output = usize;

            fn index(&self, index: usize) -> &Self::Output {
                if index >= self.dim.len() {
                    panic!("Index out of range");
                }
                &self.dim[index]
            }
        }

        impl IndexMut<usize> for $name {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                if index >= self.dim.len() {
                    panic!("Index out of range");
                }
                &mut self.dim[index]
            }
        }

        impl IntoIterator for $name {
            type Item = usize;
            type IntoIter = std::vec::IntoIter<Self::Item>;

            #[allow(clippy::unnecessary_to_owned)]
            fn into_iter(self) -> Self::IntoIter {
                self.dim.to_vec().into_iter()
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.dim == other.dim
            }
        }

        impl DimTrait for $name {
            fn len(&self) -> usize {
                self.dim.len()
            }

            fn is_empty(&self) -> bool {
                todo!();
            }
        }

        // impl IndexTrait for $name {
        //     type Dim = Self;
        //     fn offset(&self, shape: &Self::Dim, stride: &Self::Dim) -> usize {
        //         if shape.is_overflow(*self) {
        //             panic!("Dimension mismatch");
        //         }
        //
        //         self.into_iter()
        //             .zip(stride.into_iter())
        //             .map(|(x, y)| x * y)
        //             .sum()
        //     }
        // }
    };
}

impl_dim!(Dim1, [usize; 1]);
impl_dim!(Dim2, [usize; 2]);
impl_dim!(Dim3, [usize; 3]);
impl_dim!(Dim4, [usize; 4]);

macro_rules! impl_less_dim {
    ($impl_ty:ty, $less_dim:ty) => {
        impl LessDimTrait for $impl_ty {
            type LessDim = $less_dim;
        }
    };
}
impl_less_dim!(Dim1, Dim0);
impl_less_dim!(Dim2, Dim1);
impl_less_dim!(Dim3, Dim2);
impl_less_dim!(Dim4, Dim3);

macro_rules! impl_grater_dim_trait {
    ($impl_ty:ty, $less_dim:ty) => {
        impl GreaterDimTrait for $impl_ty {
            type GreaterDim = $less_dim;
        }
    };
}
impl_grater_dim_trait!(Dim0, Dim1);
impl_grater_dim_trait!(Dim1, Dim2);
impl_grater_dim_trait!(Dim2, Dim3);
impl_grater_dim_trait!(Dim3, Dim4);

#[macro_export]
macro_rules! dim {
    () => {
        $crate::dim_impl::Dim0::new()
    };
    ($x:expr) => {
        $crate::dim_impl::Dim1::new([$x])
    };
    ($x:expr, $y:expr) => {
        $crate::dim_impl::Dim2::new([$x, $y])
    };
    ($x:expr, $y:expr, $z:expr) => {
        $crate::dim_impl::Dim3::new([$x, $y, $z])
    };
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        $crate::dim_impl::Dim4::new([$x, $y, $z, $w])
    };
}

#[cfg(test)]
mod dim {
    use super::*;

    #[test]
    fn test_0d() {
        let ans = Dim0::new();
        let dim = dim!();
        assert_eq!(ans, dim);
    }

    #[test]
    fn test_1d() {
        let ans = Dim1::new([1]);
        let dim = dim!(1);
        assert_eq!(ans, dim);
    }

    #[test]
    fn test_2d() {
        let ans = Dim2::new([1, 2]);
        let dim = dim!(1, 2);
        assert_eq!(ans, dim);
    }

    #[test]
    fn test_3d() {
        let ans = Dim3::new([1, 2, 3]);
        let dim = dim!(1, 2, 3);
        assert_eq!(ans, dim);
    }

    #[test]
    fn test_4d() {
        let ans = Dim4::new([1, 2, 3, 4]);
        let dim = dim!(1, 2, 3, 4);
        assert_eq!(ans, dim);
    }
}

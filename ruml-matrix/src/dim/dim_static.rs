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
        index.len() != 0
    }

    fn slice(&self) -> &[usize] {
        &[]
    }
}

impl From<&[usize]> for Dim0 {
    fn from(dim: &[usize]) -> Self {
        if dim.len() != 0 {
            panic!("Invalid dimension");
        }
        Dim0 {}
    }
}

impl From<&[usize; 0]> for Dim0 {
    fn from(dim: &[usize; 0]) -> Self {
        if dim.len() != 0 {
            panic!("Invalid dimension");
        }
        Dim0 {}
    }
}

impl From<[usize; 0]> for Dim0 {
    fn from(dim: [usize; 0]) -> Self {
        if dim.len() != 0 {
            panic!("Invalid dimension");
        }
        Dim0 {}
    }
}

macro_rules! impl_dim {
    ($name:ident, $number_of_elm:expr) => {
        #[derive(Clone, Debug, Copy, Default)]
        pub struct $name {
            dim: [usize; $number_of_elm],
        }

        impl $name {
            pub fn new(dim: [usize; $number_of_elm]) -> Self {
                Self { dim }
            }

            pub fn dim(&self) -> &[usize; $number_of_elm] {
                &self.dim
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
            type IntoIter = std::array::IntoIter<Self::Item, $number_of_elm>;

            fn into_iter(self) -> Self::IntoIter {
                self.dim.into_iter()
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
                self.dim.iter().all(|&d| d == 0)
            }

            fn slice(&self) -> &[usize] {
                &self.dim
            }
        }

        impl From<&[usize]> for $name {
            fn from(slice: &[usize]) -> Self {
                let mut array = [0; $number_of_elm];
                array[..slice.len()].copy_from_slice(slice);
                $name { dim: array }
            }
        }

        impl From<&[usize; $number_of_elm]> for $name {
            fn from(slice: &[usize; $number_of_elm]) -> Self {
                let mut array = [0; $number_of_elm];
                array.copy_from_slice(slice);
                $name { dim: array }
            }
        }

        impl From<[usize; $number_of_elm]> for $name {
            fn from(array: [usize; $number_of_elm]) -> Self {
                $name { dim: array }
            }
        }
    };
}

impl_dim!(Dim1, 1);
impl_dim!(Dim2, 2);
impl_dim!(Dim3, 3);
impl_dim!(Dim4, 4);

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

// #[cfg(test)]
// mod dim {
//     use super::*;
//     use crate::dim;
//
//     #[test]
//     fn test_0d() {
//         let ans = Dim0::new();
//         let dim = dim!();
//         assert_eq!(ans, dim);
//     }
//
//     #[test]
//     fn test_1d() {
//         let ans = Dim1::new([1]);
//         let dim = [1];
//         assert_eq!(ans, dim);
//     }
//
//     #[test]
//     fn test_2d() {
//         let ans = Dim2::new([1, 2]);
//         let dim = [1, 2];
//         assert_eq!(ans, dim);
//     }
//
//     #[test]
//     fn test_3d() {
//         let ans = Dim3::new([1, 2, 3]);
//         let dim = [1, 2, 3];
//         assert_eq!(ans, dim);
//     }
//
//     #[test]
//     fn test_4d() {
//         let ans = Dim4::new([1, 2, 3, 4]);
//         let dim = [1, 2, 3, 4];
//         assert_eq!(ans, dim);
//     }
// }

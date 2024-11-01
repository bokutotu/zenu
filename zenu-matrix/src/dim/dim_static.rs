use std::ops::{Index, IndexMut};

use crate::dim::{DimTrait, GreaterDimTrait, LessDimTrait};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Copy, Default)]
pub struct Dim0 {}

impl Dim0 {
    #[must_use]
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

impl IntoIterator for Dim0 {
    type Item = usize;
    type IntoIter = std::array::IntoIter<Self::Item, 0>;

    fn into_iter(self) -> Self::IntoIter {
        [].into_iter()
    }
}

impl DimTrait for Dim0 {
    fn len(&self) -> usize {
        0
    }

    fn is_empty(&self) -> bool {
        true
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
        assert!(dim.is_empty(), "Invalid dimension");
        Dim0 {}
    }
}

impl From<&[usize; 0]> for Dim0 {
    fn from(dim: &[usize; 0]) -> Self {
        assert!(dim.is_empty(), "Invalid dimension");
        Dim0 {}
    }
}

impl From<[usize; 0]> for Dim0 {
    fn from(dim: [usize; 0]) -> Self {
        assert!(dim.is_empty(), "Invalid dimension");
        Dim0 {}
    }
}

impl From<&Dim0> for Dim0 {
    fn from(dim: &Dim0) -> Self {
        *dim
    }
}

macro_rules! impl_dim {
    ($name:ident, $number_of_elm:expr) => {
        #[derive(Clone, Debug, Copy, Default, Serialize, Deserialize)]
        pub struct $name {
            dim: [usize; $number_of_elm],
        }

        impl $name {
            #[must_use]
            pub fn new(dim: [usize; $number_of_elm]) -> Self {
                Self { dim }
            }

            #[must_use]
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

        impl From<&$name> for $name {
            fn from(dim: &$name) -> Self {
                *dim
            }
        }
    };
}

impl_dim!(Dim1, 1);
impl_dim!(Dim2, 2);
impl_dim!(Dim3, 3);
impl_dim!(Dim4, 4);
impl_dim!(Dim5, 5);
impl_dim!(Dim6, 6);

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
impl_less_dim!(Dim5, Dim4);
impl_less_dim!(Dim6, Dim5);

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
impl_grater_dim_trait!(Dim4, Dim5);
impl_grater_dim_trait!(Dim5, Dim6);

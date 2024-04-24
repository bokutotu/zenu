// pub mod dot;
pub mod gemm;
// pub mod gemv;
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlasTrans {
    None,
    Ordinary,
    Conjugate,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlasLayout {
    RowMajor,
    ColMajor,
}

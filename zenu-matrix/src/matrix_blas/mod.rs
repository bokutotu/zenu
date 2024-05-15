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

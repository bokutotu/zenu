pub trait TBool: Default {
    fn as_bool() -> bool;
}

#[derive(Default)]
pub struct TTrue;
#[derive(Default)]
pub struct TFalse;

impl TBool for TTrue {
    fn as_bool() -> bool {
        true
    }
}

impl TBool for TFalse {
    fn as_bool() -> bool {
        false
    }
}

pub trait TAnd<RHS: TBool> {
    type Output: TBool;
}

impl<RHS: TBool> TAnd<RHS> for TFalse {
    type Output = TFalse;
}

impl<RHS: TBool> TAnd<RHS> for TTrue {
    type Output = RHS;
}

pub trait TNat: Default {
    type IsZero: TBool;

    fn as_int(&self) -> isize;
}

#[derive(Default)]
pub struct TZero;

impl TNat for TZero {
    type IsZero = TTrue;

    fn as_int(&self) -> isize {
        0
    }
}

#[derive(Default)]
pub struct TSucc<N: TNat>(N);

impl<N: TNat> TNat for TSucc<N> {
    type IsZero = TFalse;

    fn as_int(&self) -> isize {
        self.0.as_int() + 1
    }
}

pub type TOne = TSucc<TZero>;
pub type TTwo = TSucc<TOne>;
pub type TThree = TSucc<TTwo>;
pub type TFour = TSucc<TThree>;
pub type TFive = TSucc<TFour>;

pub trait TAdd<RHS: TNat>: TNat {
    type Result: TNat;
}

impl<RHS: TNat> TAdd<RHS> for TZero {
    type Result = RHS;
}

impl<RHS: TNat, LHS: TNat + TAdd<RHS>> TAdd<RHS> for TSucc<LHS> {
    type Result = TSucc<<LHS as TAdd<RHS>>::Result>;
}

pub trait TSub<RHS: TNat>: TNat {
    type Output: TNat;
    type IsZero: TBool;
}

impl<LHS: TNat> TSub<TZero> for LHS {
    type Output = LHS;
    type IsZero = <LHS as TNat>::IsZero;
}

impl<N: TNat> TSub<TSucc<N>> for TZero {
    type Output = TZero;
    type IsZero = TTrue;
}

impl<N: TNat, M: TSub<N>> TSub<TSucc<N>> for TSucc<M> {
    type Output = <M as TSub<N>>::Output;
    type IsZero = <<M as TSub<N>>::Output as TNat>::IsZero;
}

pub trait TEqual<RHS: TNat> {
    type Output: TBool;
}

impl<N: TNat, M: TNat, Out1: TBool, Out2: TBool> TEqual<N> for M
where
    N: TSub<M, IsZero = Out1>,
    M: TSub<N, IsZero = Out2>,
    Out1: TAnd<Out2>,
{
    type Output = <Out1 as TAnd<Out2>>::Output;
}

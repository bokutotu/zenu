use std::marker::PhantomData;

use crate::{element_wise::ElementWise, num::Num};

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuElementWise<T: Num> {
    _phantom: PhantomData<T>,
}

impl<T: Num> ElementWise<T> for CpuElementWise<T> {
    fn mul(
        res: *mut T,
        lhs: *const T,
        rhs: *const T,
        size: usize,
        inc_lhs: usize,
        inc_rhs: usize,
        inc_self: usize,
    ) {
        unsafe {
            let mut res = res;
            let mut lhs = lhs;
            let mut rhs = rhs;
            for _ in 0..size {
                *res = *lhs * *rhs;
                res = res.add(inc_self);
                lhs = lhs.add(inc_lhs);
                rhs = rhs.add(inc_rhs);
            }
        }
    }
}

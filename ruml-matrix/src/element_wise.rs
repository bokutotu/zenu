pub trait ElementWise<T> {
    fn mul(
        res: *mut T,
        lhs: *const T,
        rhs: *const T,
        size: usize,
        inc_lhs: usize,
        inc_rhs: usize,
        inc_self: usize,
    );
}

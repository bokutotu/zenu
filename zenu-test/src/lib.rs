/// Don't use  this macro in sub, abs, asum, zeros
/// becase this macro uses them.
#[macro_export]
macro_rules! assert_mat_eq_epsilon {
    ($mat:expr, $mat2: expr, $epsilon:expr) => {{
        let mat = $mat;
        let mat2 = $mat2;
        let epsilon = $epsilon;
        let diff = mat.to_ref() - mat2.to_ref();
        let abs = diff.abs();
        let diff_asum = abs.asum();
        if diff_asum > epsilon {
            panic!(
                "assertion failed: `(left == right)`\n\
                left: \n{:?},\n\
                right: \n{:?}\n\
                diff: \n{:?}\n\
                diff_asum: \n{:?}",
                mat, mat2, diff, diff_asum
            );
        }
    }};
}

#[macro_export]
macro_rules! assert_val_eq_epsilon {
    ($val:expr, $mat:expr, $grad:expr, $epsilon:expr) => {
        let val = $val;
        let mat = $mat;
        let grad = $grad;
        let epsilon = $epsilon;
        let data = val.get_data().to_ref();
        let grad = val.get_grad().unwrap().get_data().to_ref();
        zenu_test::assert_mat_eq_epsilon!(data, mat, epsilon);
        zenu_test::assert_mat_eq_epsilon!(grad, grad, epsilon);
    };
}

#[macro_export]
macro_rules! assert_val_eq {
    ($val:expr, $mat:expr, $epsilon:expr) => {
        let val = $val;
        let mat = $mat;
        let epsilon = $epsilon;
        let val_data = val.get_data();
        zenu_test::assert_mat_eq_epsilon!(val_data, mat, epsilon);
    };
}

#[macro_export]
macro_rules! assert_val_eq_grad {
    ($val:expr, $grad:expr, $epsilon:expr) => {
        let val = $val;
        let grad = $grad;
        let epsilon = $epsilon;
        let val_grad = val.get_grad().unwrap();
        let val_grad = val_grad.get_data();
        zenu_test::assert_mat_eq_epsilon!(val_grad, grad, epsilon);
    };
}

#[macro_export]
macro_rules! run_test {
    ($test_func:ident, $cpu_name:ident, $gpu_name:ident) => {
        #[test]
        fn $cpu_name() {
            $test_func::<zenu_matrix::device::cpu::Cpu>();
        }
        #[cfg(feature = "nvidia")]
        #[test]
        fn $gpu_name() {
            $test_func::<zenu_matrix::device::nvidia::Nvidia>();
        }
    };
}

#[allow(clippy::crate_in_macro_def)]
#[macro_export]
macro_rules! run_mat_test {
    ($test_func:ident, $cpu_name:ident, $gpu_name:ident) => {
        #[test]
        fn $cpu_name() {
            $test_func::<crate::device::cpu::Cpu>();
        }
        #[cfg(feature = "nvidia")]
        #[test]
        fn $gpu_name() {
            $test_func::<crate::device::nvidia::Nvidia>();
        }
    };
}

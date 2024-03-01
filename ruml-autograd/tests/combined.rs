use ruml_autograd::Variable;
use ruml_matrix::matrix::{AsPtr, ToViewMatrix};

fn formula(x: Variable<f32>, y: Variable<f32>, z: Variable<f32>) -> Variable<f32> {
    let a = x * y;
    let b = a + z;
    b
}

#[test]
fn test_formula() {
    let x = Variable::from(3.0);
    let y = Variable::from(4.0);
    let z = Variable::from(5.0);
    let result = formula(x.clone(), y.clone(), z.clone());
    assert_eq!(unsafe { *result.get_data().to_view().as_ptr() }, 17.0);
    result.backward();
    x.with_grad_data(|grad| {
        assert_eq!(unsafe { *grad.to_view().as_ptr() }, 4.0);
    });
    y.with_grad_data(|grad| {
        assert_eq!(unsafe { *grad.to_view().as_ptr() }, 3.0);
    });
    z.with_grad_data(|grad| {
        assert_eq!(unsafe { *grad.to_view().as_ptr() }, 1.0);
    });
}

fn use_twice(x: Variable<f32>) -> Variable<f32> {
    let a = x.clone() * x;
    let b = a.clone() + Variable::from(3.0);
    b
}

#[test]
fn test_use_twice() {
    let x = Variable::from(3.0);
    let result = use_twice(x.clone());
    assert_eq!(unsafe { *result.get_data().to_view().as_ptr() }, 12.0);
    result.backward();
    x.with_grad_data(|grad| {
        assert_eq!(unsafe { *grad.to_view().as_ptr() }, 3.0);
    });
}

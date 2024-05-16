// use zenu_autograd::Variable;
//
// fn test_function() -> Variable<f64> {
//     let a = Variable::from(1.0);
//     let b = Variable::from(2.0);
//
//     b.set_is_train(true);
//     b.set_name("b");
//
//     let c = Variable::from(3.0);
//     c.set_name("c");
//     c.set_is_train(true);
//
//     a * b + c
// }
//
// #[test]
// fn all_trainable() {
//     let a = test_function();
//     let variables = a.get_all_trainable_variables();
//     assert_eq!(variables.len(), 2);
//     let mut names = variables
//         .iter()
//         .map(|v| v.get_name().unwrap())
//         .collect::<Vec<_>>();
//     names.sort();
//     assert_eq!(names, vec!["b", "c"]);
// }
//
// fn more_complicated(a: Variable<f32>, b: Variable<f32>, c: Variable<f32>) -> Variable<f32> {
//     let d = a * b;
//     let e = d.clone() + c;
//     let f = e * d;
//     f
// }
//
// #[test]
// fn copilicated() {
//     let a = Variable::from(1.0);
//     a.set_name("a");
//     a.set_is_train(true);
//
//     let c = more_complicated(a.clone(), a.clone(), a.clone());
//     let d = more_complicated(c.clone(), a.clone(), a.clone());
//     let variables = d.get_all_trainable_variables();
//     assert_eq!(variables.len(), 1);
//     assert_eq!(variables[0].get_name().unwrap(), "a");
// }
//
// #[test]
// fn ultra_large_copilicated() {
//     let a = Variable::from(1.0);
//     a.set_name("a");
//     a.set_is_train(true);
//
//     let mut c = more_complicated(a.clone(), a.clone(), a.clone());
//     for _ in 0..500 {
//         c = more_complicated(c.clone(), a.clone(), a.clone());
//     }
//     let variables = c.get_all_trainable_variables();
//     assert_eq!(variables.len(), 1);
// }

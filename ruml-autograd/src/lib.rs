use std::{
    borrow::BorrowMut,
    cell::{Ref, RefCell, RefMut},
    ops::{Deref, DerefMut},
    rc::{Rc, Weak},
};

pub trait Node {
    fn inputs(&self) -> Vec<VariableWeak>;
    fn outputs(&self) -> Vec<VariableWeak>;
    fn forward(&self);
    fn backward(&self) -> Box<dyn Node>;
}

pub struct Graph {
    variable: Vec<Rc<RefCell<VariableInternal>>>,
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            variable: Vec::new(),
        }
    }

    pub fn add_variable(&mut self, value: VariableStrong) {
        self.variable.push(value.internal);
    }

    pub fn mearge(left: &Self, right: &Self) -> Self {
        let mut variable = left.variable.clone();
        variable.extend(right.variable.clone().into_iter());
        Graph {
            variable: variable.to_vec(),
        }
    }
}

pub struct VariableInternal {
    value: Rc<RefCell<f64>>,
    parent: Vec<Rc<RefCell<Box<dyn Node>>>>,
    children: Vec<Rc<RefCell<Box<dyn Node>>>>,
    graph: Rc<RefCell<Graph>>,
}

impl VariableInternal {
    pub fn new(value: f64) -> VariableInternal {
        let value = Rc::new(RefCell::new(value));
        VariableInternal {
            value,
            parent: Vec::new(),
            children: Vec::new(),
            graph: Rc::new(RefCell::new(Graph::new())),
        }
    }

    pub fn set_value(&mut self, value: f64) {
        self.value.borrow_mut().replace(value);
    }

    pub fn get_parent(&self) -> Vec<Rc<RefCell<Box<dyn Node>>>> {
        self.parent.clone()
    }

    pub fn get_children(&self) -> Vec<Rc<RefCell<Box<dyn Node>>>> {
        self.children.clone()
    }

    pub fn add_child(&mut self, child: Rc<RefCell<Box<dyn Node>>>) {
        self.children.push(child);
    }

    pub fn add_parent(&mut self, parent: Rc<RefCell<Box<dyn Node>>>) {
        self.parent.push(parent);
    }

    pub fn get_graph(&self) -> Rc<RefCell<Graph>> {
        self.graph.clone()
    }

    pub fn get_value(&self) -> f64 {
        *self.value.borrow()
    }
}

#[derive(Clone)]
pub struct VariableStrong {
    internal: Rc<RefCell<VariableInternal>>,
}

impl VariableStrong {
    pub fn new(value: f64) -> VariableStrong {
        VariableStrong {
            internal: Rc::new(RefCell::new(VariableInternal::new(value))),
        }
    }

    fn borrow<'a>(&'a self) -> Ref<'a, VariableInternal> {
        (*self.internal).borrow()
    }

    fn borrow_mut<'a>(&'a self) -> RefMut<'a, VariableInternal> {
        (*self.internal).borrow_mut()
    }

    fn downgrade(&self) -> VariableWeak {
        VariableWeak {
            internal: Rc::downgrade(&self.internal),
        }
    }
}

#[derive(Clone)]
pub struct VariableWeak {
    internal: Weak<RefCell<VariableInternal>>,
}

impl VariableWeak {
    fn upgrade(&self) -> Option<Rc<RefCell<VariableInternal>>> {
        self.internal.upgrade()
    }
}

pub struct Add {
    pub x: VariableWeak,
    pub y: VariableWeak,
    pub output: VariableWeak,
}

impl Node for Add {
    fn forward(&self) {
        let x_val = (*self.x.upgrade().unwrap()).borrow().get_value();
        let y_val = (*self.y.upgrade().unwrap()).borrow().get_value();

        (*self.output.upgrade().unwrap())
            .borrow_mut()
            .set_value(x_val + y_val);
    }

    fn inputs(&self) -> Vec<VariableWeak> {
        vec![self.x.clone(), self.y.clone()]
    }

    fn outputs(&self) -> Vec<VariableWeak> {
        vec![self.output.clone()]
    }

    fn backward(&self) -> Box<dyn Node> {
        let x_grad = VariableInternal::new(1.0);
        let y_grad = VariableInternal::new(1.0);
        todo!();
    }
}

pub fn make_graph_2input_1output(
    x: VariableStrong,
    y: VariableStrong,
    output: VariableStrong,
    node: Rc<RefCell<Box<dyn Node>>>,
) {
    let mut mearged_graph = Graph::mearge(
        &(*x.borrow().get_graph().borrow()),
        &(*y.borrow().get_graph().borrow()),
    );
    mearged_graph.add_variable(output.clone());
    x.borrow_mut().add_child(node.clone());
    y.borrow_mut().add_child(node.clone());
    output.borrow_mut().add_parent(node.clone());
    output.borrow_mut().graph.replace(mearged_graph);
    (*node).borrow().forward();
}

impl Add {
    fn new(x: VariableStrong, y: VariableStrong, output: VariableStrong) -> Self {
        Add {
            x: x.downgrade(),
            y: y.downgrade(),
            output: output.downgrade(),
        }
    }

    pub fn add_variable(
        x: VariableStrong,
        y: VariableStrong,
        output: VariableStrong,
    ) -> VariableStrong {
        let add = Add::new(x.clone(), y.clone(), output.clone());
        make_graph_2input_1output(x, y, output.clone(), Rc::new(RefCell::new(Box::new(add))));
        output
    }
}

pub fn add(x: VariableStrong, y: VariableStrong) -> VariableStrong {
    let output = VariableStrong::new(0.0);
    Add::add_variable(x, y, output)
}

pub fn val(x: f64) -> VariableStrong {
    VariableStrong::new(x)
}

#[cfg(test)]
mod autograd {
    use super::*;
    #[test]
    fn test_add() {
        let x = val(2.0);
        let y = val(3.0);
        let z = add(x.clone(), y.clone());
        assert_eq!(z.borrow().get_value(), 5.0);
    }
}
